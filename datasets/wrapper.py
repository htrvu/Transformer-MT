import torch
import torchtext.legacy.data as data
import os
import pandas as pd
from datasets.iterator import MyIterator
from datasets.tokenizer import Tokenizer
from constants import *
from datasets.utils import read_data
import dill

global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    The batch size is equivalent to the number of tokens in the examples.

    Args:
        - new (dict): dictionary about the new example to add to current batch
        - count (int): number of examples in current batch
        - sofar (int): number of tokens in current batch (current effective batch size)
        
    Returns: new effective batch size
    """

    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class TextDataWrapper:
    """
    Data wrapper for text dataset
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

        self.__create_fields()


    def __create_fields(self):
        """
        Create fields for given source and target languages
        """
        print("Loading tokenizers...")
        src_tokenizer = Tokenizer(self.src_lang)
        trg_tokenizer = Tokenizer(self.trg_lang)

        print("Creating fields...")
        self.src_field = data.Field(lower=True, tokenize=src_tokenizer.tokenize)
        self.trg_field = data.Field(
            lower=True,
            tokenize=trg_tokenizer.tokenize,
            init_token=SOS,
            eos_token=EOS,
        )
        print('Done')
        print('------------------------')


    def create_dataloader(self, src_path: str, trg_path: str, is_train: bool = True) -> data.Iterator:
        """
        Create dataset and iterator

        Args:
            src_data (List[str]): source data
            trg_data (List[str]): target data
            is_train (bool, optional): identify train/val split. Defaults to True.
        """
        print("Reading source and target data from files:", src_path, trg_path)
        src_data, trg_data = read_data(src_path, trg_path)

        print('Building dataloader...')

        raw_data = {
            "src": [line for line in src_data],
            "trg": [line for line in trg_data],
        }
        df = pd.DataFrame(raw_data, columns=["src", "trg"])
        mask = (df["src"].str.count(" ") < self.max_len) & (
            df["trg"].str.count(" ") < self.max_len
        )
        df = df.loc[mask]   # Remove source and target text that are too long
        df.to_csv("temp.csv", index=False)

        data_fields = [("src", self.src_field), ("trg", self.trg_field)]
        dataset = data.TabularDataset("./temp.csv", format='csv', fields=data_fields)
        dataloader = MyIterator(
            dataset,
            batch_size=self.batch_size,
            device=self.device,
            repeat=False,
            sort_key=lambda x: (len(x.src), len(x.trg)),
            shuffle=True,
            train=is_train,
            batch_size_fn=batch_size_fn,
        )

        os.remove('temp.csv')

        # Only build vocab based on train data
        if is_train:
            self.src_field.build_vocab(dataset, )
            self.trg_field.build_vocab(dataset)
            print('Saving fields...')
            dst_root = './fields'
            os.makedirs(dst_root, exist_ok=True)
            torch.save(self.src_field, os.path.join(dst_root, 'src_field.pt'), pickle_module=dill)
            torch.save(self.trg_field, os.path.join(dst_root, 'trg_field.pt'), pickle_module=dill)

        print('Done')
        print('------------------------')

        return dataloader


if __name__=="__main__":
    data_wrapper = TextDataWrapper(src_lang="en_core_web_sm", trg_lang="vi_core_news_lg", max_len=160, batch_size=128, device="cuda")

    train_iter = data_wrapper.create_dataloader(src_path="data/train.en", trg_path="data/train.vi", is_train=True)
    valid_iter = data_wrapper.create_dataloader(src_path="data/tst2013.en", trg_path="data/tst2013.vi", is_train=False)

    # vocab size
    print(len(data_wrapper.src_field.vocab))
    print(len(data_wrapper.trg_field.vocab))

    print(data_wrapper.src_field.vocab.stoi['<pad>'])
    print(data_wrapper.trg_field.vocab.stoi['<pad>'])