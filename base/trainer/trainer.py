import os

import torch
import torch.nn as nn
from torchtext.legacy.data import Field, Batch, Iterator
import time
from typing import Callable, Type, Dict

from base.metrics import calc_bleu
from base.predictor import Predictor


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

        trg_input = trg[:, :-1]

        preds, _, _ = self.model(src, trg_input)

        ys = trg[:, 1:].contiguous().view(-1)

        self.optimizer.zero_grad()
        loss = self.criterion(preds.view(-1, preds.size(-1)), ys)
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

            trg_input = trg[:, :-1]

            preds, _, _ = self.model(src, trg_input)

            ys = trg[:, 1:].contiguous().view(-1)

            loss = self.criterion(preds.view(-1, preds.size(-1)), ys)

            total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)

        return avg_loss

    def fit(self, train_iter: Iterator, valid_iter: Iterator, beam_size: int = 1, ckpt_folder: str = './weights', log_interval: int = 50) -> None:
        """
        Fit model

        Args:
            train_iter (Iterator): Train iterator
            valid_iter (Iterator): Valid iterator
            k (int): Beam size
        """
        print('Start training...')
        print('The checkpoints will be saved at', ckpt_folder)
        os.makedirs(ckpt_folder, exist_ok=True)

        min_valid_loss = float('inf')
        for epoch in range(self.num_epochs):
            # Train
            total_loss = 0
            s = time.time()
            for i, batch in enumerate(train_iter):
                loss = self.step(batch)
                total_loss += loss.item()

                if i % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    print(
                        f"Epoch: {epoch + 1:3}/{self.num_epochs} | Time: {time.time() - s:.2f}s | Loss: {avg_loss:.4f}",
                        end='\r'
                    )
                    total_loss = 0

            # Validate
            s = time.time()
            valid_loss = self.validate(valid_iter)
            print(
                f"Epoch: {epoch + 1:3}/{self.num_epochs} | Time: {time.time() - s:.2f}s | Train Loss: {avg_loss:.4f} | Valid Loss: {valid_loss:.4f}"
            )

            # Save checkpoint
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'{ckpt_folder}/best.pt')
            torch.save(self.model.state_dict(), f'{ckpt_folder}/last.pt')


        # Only calculate BLEU on first 500 sentences of validation set
        print(f'Evaluating BLEU score with beam size = {beam_size}...')
        val_src = [' '.join(x.src) for x in valid_iter.dataset.examples[:10]][1:]
        val_trg = [' '.join(x.trg) for x in valid_iter.dataset.examples[:10]][1:]
        predictor = Predictor(self.model, self.src_field, self.trg_field, self.device)
        pred_sentences = []
        for sentence in val_src:
            pred_trg = predictor(sentence, self.max_len, beam_size)
            pred_sentences.append(pred_trg)

        pred_sentences = [self.trg_field.preprocess(sentence) for sentence in pred_sentences]
        trg_sentences = [[sent.split()] for sent in val_trg]
        bleu_score = self.metric(pred_sentences, trg_sentences)

        print(f"BLEU score: {bleu_score:.4f}")