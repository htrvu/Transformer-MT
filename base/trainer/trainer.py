import os

import json
import torch
import torch.nn as nn
from torchtext.legacy.data import Field, Batch, Iterator
import matplotlib.pyplot as plt
import time
from typing import Callable, Type

from base.predictor import Predictor
from transformer.helpers import gen_mask
from constants import *


class Trainer:
    """
    Trainer wrapper class
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Type,
        criterion: nn.Module,
        num_epochs: int,
        metric: Callable,
        src_field: Field,
        trg_field: Field,
        max_len: int,
        device: str = "cuda:0",
    ) -> None:
        """
        Args:
            - model (nn.Module): Model to be trained
            - optimizer (Type): Optimizer to be used
            - criterion (nn.Module): Loss function
            - num_epochs (int): Number of epochs
            - metric (Callable): Metric function
            - src_field (Field): Tokenizer for source language
            - trg_field (Field): Tokenizer for target language
            - max_len (int): Maximum length of sentence
            - device (str): Device to run model (e.g. cuda:0, cpu)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.metric = metric

        self.src_field = src_field
        self.trg_field = trg_field
        self.max_len = max_len

        self.device = device

        self.model.to(self.device)

    def step(self, batch: Batch) -> nn.Module:
        """
        A step of training

        Args:
            batch (Batch): Batch of data

        Returns:
            nn.Module: loss of the step
        """
        self.model.train()

        src = batch.src.transpose(0, 1).to(self.device)
        trg = batch.trg.transpose(0, 1).to(self.device)
        trg_inputs = trg[:, :-1]
        trg_labels = trg[:, 1:].contiguous().view(-1)

        enc_padding_mask, dec_look_ahead_mask = gen_mask(
            src,
            self.src_field.vocab.stoi[PAD],
            trg_inputs,
            self.trg_field.vocab.stoi[PAD],
        )
        preds, _, _ = self.model(src, trg_inputs, enc_padding_mask, dec_look_ahead_mask)

        self.optimizer.zero_grad()
        loss = self.criterion(preds.view(-1, preds.size(-1)), trg_labels)
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def validate(self, val_iter: Iterator) -> float:
        """
        Validate model

        Args:
            val_iter (Iterator): Validation iterator

        Returns:
            float: Average loss of validation
        """
        self.model.eval()

        total_loss = []
        for _, batch in enumerate(val_iter):
            src = batch.src.transpose(0, 1).to(self.device)
            trg = batch.trg.transpose(0, 1).to(self.device)
            trg_inputs = trg[:, :-1]
            trg_labels = trg[:, 1:].contiguous().view(-1)

            enc_padding_mask, dec_look_ahead_mask = gen_mask(
                src,
                self.src_field.vocab.stoi[PAD],
                trg_inputs,
                self.trg_field.vocab.stoi[PAD],
            )

            preds, _, _ = self.model(
                src, trg_inputs, enc_padding_mask, dec_look_ahead_mask
            )

            loss = self.criterion(preds.view(-1, preds.size(-1)), trg_labels)
            total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)

        return avg_loss

    def fit(
        self,
        train_iter: Iterator,
        valid_iter: Iterator,
        beam_size: int = 1,
        out_dir: str = "./runs",
        log_interval: int = 100,
    ) -> None:
        """
        Fit model

        Args:
            train_iter (Iterator): Train iterator
            valid_iter (Iterator): Valid iterator
            k (int): Beam size
        """
        print("Start training...")
        print("The training results will be saved at", out_dir)

        # Only calculate BLEU on first 500 sentences of validation set
        val_src = [" ".join(x.src) for x in valid_iter.dataset.examples[:500]][1:]
        val_trg = [" ".join(x.trg) for x in valid_iter.dataset.examples[:500]][1:]
        predictor = Predictor(self.model, self.src_field, self.trg_field, self.device)

        log = []
        train_losses = []
        valid_losses = []
        bleu_scores = []
        min_valid_loss = float("inf")
        for epoch in range(self.num_epochs):
            # Train
            epoch_loss = 0
            cnt = 0
            iter_loss = 0
            s = time.time()
            for i, batch in enumerate(train_iter):
                loss = self.step(batch)
                iter_loss += loss.item()
                epoch_loss += loss.item()
                cnt += 1

                if i % log_interval == 0:
                    avg_loss = iter_loss / log_interval
                    print(
                        f"Epoch: {epoch + 1:3}/{self.num_epochs} | Iteration: {i} | Loss: {avg_loss:.4f}",
                        end="\r",
                    )
                    iter_loss = 0

            avg_loss = epoch_loss / cnt

            # Validate
            valid_loss = self.validate(valid_iter)

            # BLEU score
            pred_sentences = []
            
            for sentence in val_src:
                pred_trg = predictor(sentence, self.max_len, beam_size)
                pred_sentences.append(pred_trg)

            pred_sentences = [
                self.trg_field.preprocess(sentence) for sentence in pred_sentences
            ]
            trg_sentences = [[sent.split()] for sent in val_trg]
            bleu_score = self.metric(pred_sentences, trg_sentences)

            print(
                f"Epoch: {epoch + 1:3}/{self.num_epochs} | Time: {time.time() - s:.2f}s | Train Loss: {avg_loss:.4f} | Valid Loss: {valid_loss:.4f} | BLEU score: {bleu_score}"
            )

            log.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": valid_loss,
                    "bleu_score": bleu_score,
                }
            )
            train_losses.append(avg_loss)
            valid_losses.append(valid_loss)
            bleu_scores.append(bleu_score)

            # Write log
            with open(os.path.join(out_dir, "log.json"), "w") as f:
                json.dump(log, f, indent=4)

            # Save checkpoint
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f"{out_dir}/best.pt")
            torch.save(self.model.state_dict(), f"{out_dir}/last.pt")

            # Plot: Loss
            plt.figure()
            plt.plot(train_losses, label="Train Loss")
            plt.plot(valid_losses, label="Val Loss")
            plt.legend()
            plt.xlabel("Epoch")
            plt.title("Loss")
            plt.savefig(os.path.join(out_dir, "loss_plot.png"))

            # Plot: BLEU score
            plt.figure()
            plt.plot(bleu_scores, label="BLEU score")
            plt.legend()
            plt.xlabel("Epoch")
            plt.title(f"BLEU score with beam size = {beam_size}")
            plt.savefig(os.path.join(out_dir, "bleu_plot.png"))
