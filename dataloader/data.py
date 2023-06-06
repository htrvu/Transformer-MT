import torchtext.legacy.data as data
import os
import pandas as pd

import spacy
import re

from dataloader.iterator import MyIterator, batch_size_fn

from typing import List, Tuple


class Token:
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy

        Args:
            - text (str): text to be tokenized

        Returns: List[str]: tokenized text
        """
        text = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(text))
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)
        text = text.lower()
        try:
            return [tok.text for tok in self.nlp.tokenizer(text) if tok.text != " "]
        except:
            # Strange behavior of spacy sometimes, text is blank after tokenization
            return []


class DataWrapper:
    """
    Data wrapper for dataset
    """
    def __init__(self, src_lang: str, trg_lang: str, max_len: int, batch_size: int, device: str):
        """
        Args:
            src_lang (str): language of source text
            trg_lang (str): language of target text
            max_len (int): maximum length of sentence
            batch_size (int): batch size
            device (str): device to run model
        """
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device

    def load(self, src_file: str, trg_file: str) -> Tuple[List[str], List[str]]:
        """
        Load data from file
        
        Args:
            src_file (str): source file
            trg_file (str): target file

        Returns: Tuple[List[str], List[str]]: source data and target data
        """
        src_data = open(src_file, "r", encoding="utf8").read().strip().split("\n")
        trg_data = open(trg_file, "r", encoding="utf8").read().strip().split("\n")

        return src_data, trg_data

    def create_fields(self):
        """
        Create fields for dataset using spaCy tokenizer
        """
        print("Loading spaCy tokenizers...")

        tokenized_src = Token(self.src_lang)
        tokenized_trg = Token(self.trg_lang)

        self.SRC = data.Field(lower=True, tokenize=tokenized_src.tokenizer)
        self.TRG = data.Field(
            lower=True,
            tokenize=tokenized_trg.tokenizer,
            init_token="<sos>",
            eos_token="<eos>",
        )

    def create_dataset(self, src_data, trg_data, is_train=True):
        """
        Create dataset and iterator

        Args:
            is_train (bool, optional): identify train/val split. Defaults to True.
        """
        print("Creating dataset and iterator... ")

        raw_data = {
            "src": [line for line in src_data],
            "trg": [line for line in trg_data],
        }
        df = pd.DataFrame(raw_data, columns=["src", "trg"])

        mask = (df["src"].str.count(" ") < self.max_len) & (
            df["trg"].str.count(" ") < self.max_len
        )

        df = df.loc[mask]
        df.to_csv("temp.csv", index=False)

        data_fields = [("src", self.SRC), ("trg", self.TRG)]

        train = data.TabularDataset("./temp.csv", format='csv', fields=data_fields)

        train_iter = MyIterator(
            train,
            batch_size=self.batch_size,
            device=self.device,
            repeat=False,
            sort_key=lambda x: (len(x.src), len(x.trg)),
            shuffle=True,
            train=is_train,
            batch_size_fn=batch_size_fn,
        )

        os.remove('temp.csv')

        if is_train:
            self.SRC.build_vocab(train)
            self.TRG.build_vocab(train)

        return train_iter


if __name__=="__main__":
    data_wrapper = DataWrapper(src_lang="en_core_web_sm", trg_lang="vi_core_news_lg", max_len=160, batch_size=128, device="cuda")

    train_src_data, train_trg_data = data_wrapper.load(src_file="data/train.en", trg_file="data/train.vi")
    valid_src_data, valid_trg_data = data_wrapper.load(src_file="data/tst2013.en", trg_file="data/tst2013.vi")

    data_wrapper.create_fields()

    train_iter = data_wrapper.create_dataset(train_src_data, train_trg_data, is_train=True)
    valid_iter = data_wrapper.create_dataset(valid_src_data, valid_trg_data, is_train=False)

    print(data_wrapper.SRC.vocab)
    print(data_wrapper.TRG.vocab)


    print(data_wrapper.SRC.vocab.stoi['<pad>'])
    print(data_wrapper.TRG.vocab.stoi['<pad>'])