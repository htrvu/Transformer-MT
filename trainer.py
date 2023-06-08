import torch
import torch.nn as nn

from torchtext.legacy.data import Field, Batch, Iterator

from typing import Callable, Type, Dict

from nltk.corpus import wordnet


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


def _beam_search(
    src: str, model: nn.Module, trg_field: Field, device: str, k: int, max_len: int
) -> str:
    return None


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

        sentence = torch.LongTensor(indexed).unsqueeze(0).to(self.device)

        sentence = _beam_search(
            sentence, model, self.trg_field, self.device, self.k, self.max_len
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
