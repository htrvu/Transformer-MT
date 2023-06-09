import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext.legacy.data import Field, Batch, Iterator

from typing import Callable, Type, Dict

from nltk.corpus import wordnet

from transformer.helpers import gen_mask


def get_synonym(token: str, vocab: Field) -> int:
    syns = wordnet.synsets(token)
    for syn in syns:
        for l in syn.lemmas():
            if l.name() in vocab.stoi:
                return vocab.stoi[l.name()]
    return 0


def multiple_replace(dict: Dict, text: str) -> str:
    import re

    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


def generate(
    src: Variable,
    model: nn.Module,
    src_field: Field,
    trg_field: Field,
    device: str,
    k: int,
    max_len: int,
):
    init_token = trg_field.vocab.stoi["<sos>"]
    src_mask = (src != src_field.vocab.stoi["<pad>"]).unsqueeze(-2)

    e_output = model.encoder(src, src_mask)

    outputs = torch.LongTensor([[init_token]]).to(device)

    _, trg_mask, _ = gen_mask(src, outputs)

    out = model.final_ffn(model.decoder(outputs, e_output, trg_mask, src_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([torch.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(k, max_len).long().to(device)
    outputs[:, 0] = init_token
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(k, e_output.size(-2), e_output.size(-1)).to(device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(
    outputs: Variable, out: Variable, log_scores: Variable, i: int, k: int
):
    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([torch.log(p) for p in probs.data.view(-1)]).view(
        k, -1
    ) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_scores.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def _beam_search(
    src: Variable,
    model: nn.Module,
    src_field: Field,
    trg_field: Field,
    device: str,
    k: int,
    max_len: int,
) -> Variable:
    outputs, e_outputs, log_scores = generate(
        src, model, src_field, trg_field, device, k, max_len
    )
    eos_tok = trg_field.vocab.stoi["<eos>"]
    src_mask = (src != src_field.vocab.stoi["<pad>"]).unsqueeze(-2)
    idx = None
    for i in range(2, max_len):
        _, trg_mask, _ = gen_mask(src, outputs[:, :i])
        out = model.final_ffn(
            model.decoder(outputs[:, :i], e_outputs, trg_mask, src_mask)
        )
        out = F.softmax(out, dim=-1)
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, k)
        ones = (
            outputs == eos_tok
        ).nonzero()  # Occurrences of end symbols for all input sentences.
        sentences_length = torch.zeros(len(outputs), dtype=torch.long).to(device)
        for vec in ones:
            i = vec[0]
            if sentences_length[i] == 0:  # First end symbol has not been found yet
                sentences_length[i] = vec[1]  # Position of end symbol
        num_finished_sentences = len([s for s in sentences_length if s > 0])
        if num_finished_sentences == k:
            alpha = 0.7
            div = 1 / (sentences_length.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            idx = ind.data[0]
            break

    if idx is None:
        length = (outputs[0] == eos_tok).nonzero()[0]
        return " ".join([trg_field.vocab.itos[tok] for tok in outputs[0][1:length]])
    else:
        length = (outputs[idx] == eos_tok).nonzero()[0]
        return " ".join([trg_field.vocab.itos[tok] for tok in outputs[idx][1:length]])


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
        scorer: Callable,
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
            - scorer (Callable): Scorer function
            - src_field (Field): Tokenizer for source language
            - trg_field (Field): Tokenizer for target language
            - max_len (int): Maximum length of sentence
            - device (str): Device to run model (e.g. cuda:0, cpu)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.scorer = scorer

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

        print(src.shape, trg.shape)
        print(src, trg)

        trg_input = trg[:, :-1]

        # encoder_mask, decoder_mask, cross_mask = gen_mask(src, trg_input)

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

            preds = self.model(src, trg_input)

            ys = trg[:, 1:].contiguous().view(-1)

            loss = self.criterion(preds.view(-1, preds.size(-1)), ys)

            total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)

        return avg_loss

    def fit(self, train_iter: Iterator, valid_iter: Iterator, k: int) -> None:
        """
        Fit model

        Args:
            train_iter (Iterator): Train iterator
            valid_iter (Iterator): Valid iterator
            k (int): Beam size
        """
        import time

        for epoch in range(self.num_epochs):
            total_loss = 0

            for i, batch in enumerate(train_iter):
                s = time.time()

                loss = self.step(batch)

                total_loss += loss.item()

                if i % 100 == 0:
                    avg_loss = total_loss / 100
                    print(
                        f"Epoch: {epoch + 1} | Time: {time.time() - s:.2f}s | Loss: {avg_loss:.4f}"
                    )
                    total_loss = 0

            s = time.time()

            valid_loss = self.validate(valid_iter)

            bleu_score = self.scorer(
                valid_src_data[:500],
                valid_trg_data[:500],
                model,
                self.src_field,
                self.trg_field,
                self.device,
                k,
                self.max_len,
                callback=self.predict,
            )

            print(
                f"Epoch: {epoch + 1} | Time: {time.time() - s:.2f}s | Valid Loss: {valid_loss:.4f} | BLEU sore: {bleu_score:.4f}"
            )

    def predict(self, sentence: str) -> str:
        self.model.eval()

        indexed = []

        sentence = self.src_field.preprocess(sentence)

        for token in sentence:
            if self.src_field.vocab.stoi[token] != self.src_field.vocab.stoi["<eos>"]:
                indexed.append(self.src_field.vocab.stoi[token])
            else:
                indexed.append(get_synonym(token, self.src_field.vocab))

        sentence = Variable(torch.LongTensor(indexed).unsqueeze(0).to(self.device))

        sentence = _beam_search(
            sentence,
            model,
            self.src_field,
            self.trg_field,
            self.device,
            self.k,
            self.max_len,
        )

        return multiple_replace(
            {" ?": "?", " !": "!", " .": ".", "' ": "'", " ,": ","}, sentence
        )


if __name__ == "__main__":
    from base.losses.loss import LabelSmoothingLoss
    from base.metrics.bleu import bleu
    from base.schedulers.lr_scheduler import LearningRateScheduler
    from transformer.model import Transformer
    from dataloader.data import DataWrapper

    model = Transformer(config_path="./configs/_base_.yaml")

    # Init weights for models
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    num_epochs = 10
    scorer = bleu

    data_wrapper = DataWrapper(
        src_lang="en_core_web_sm",
        trg_lang="vi_core_news_lg",
        max_len=160,
        batch_size=128,
        device="cpu",
    )

    train_src_data, train_trg_data = data_wrapper.load(
        src_file="data/train.en", trg_file="data/train.vi"
    )
    valid_src_data, valid_trg_data = data_wrapper.load(
        src_file="data/tst2013.en", trg_file="data/tst2013.vi"
    )

    data_wrapper.create_fields()

    train_iter = data_wrapper.create_dataset(
        train_src_data, train_trg_data, is_train=True
    )
    valid_iter = data_wrapper.create_dataset(
        valid_src_data, valid_trg_data, is_train=False
    )

    optimizer = LearningRateScheduler(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
        init_lr=0.2,
        d_model=512,
        n_warmup_steps=1000,
    )
    criterion = LabelSmoothingLoss(
        classes=data_wrapper.TRG.vocab.__len__(),
        padding_idx=data_wrapper.TRG.vocab.stoi["<pad>"],
        smoothing=0.1,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        scorer=scorer,
        src_field=data_wrapper.SRC,
        trg_field=data_wrapper.TRG,
        max_len=160,
        device="cpu",
    )
    
    trainer.fit(train_iter, valid_iter, k=5)