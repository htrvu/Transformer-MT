import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext.legacy.data import Field
from base.predictor.utils import *
from constants import *

class Predictor:
    """
    Predictor class
    """
    
    def __init__(
        self,
        model: nn.Module,
        src_field: Field,
        trg_field: Field,
        device: str = "cuda:0",
    ) -> None:
        """
        Args:
            - model (nn.Module): Model used for prediction
            - src_field (Field): Tokenizer for source language
            - trg_field (Field): Tokenizer for target language
            - device (str): Device to run model (e.g. cuda:0, cpu)
        """
        self.model = model
        self.src_field = src_field
        self.trg_field = trg_field
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, sentence: str, max_len: int, beam_size: int = 1) -> str:
        """
        Args:
            - sentence (str): Input sentence
            - max_len (int): Maximum length of output sentence
            - beam_size (int): Beam size

        Returns:
            str: Translated sentence
        """
        indexed = []

        sentence = self.src_field.preprocess(sentence)

        for token in sentence:
            if self.src_field.vocab.stoi[token] != self.src_field.vocab.stoi[EOS]:
                indexed.append(self.src_field.vocab.stoi[token])
            else:
                indexed.append(get_synonym(token, self.src_field))

        sentence = Variable(torch.LongTensor(indexed).unsqueeze(0).to(self.device))

        sentence = beam_search(sentence, self.model, self.src_field, self.trg_field, max_len, beam_size, self.device)
        
        print(sentence)
        return multiple_replace(
            {
                " ?": "?", 
                " !": "!", 
                " .": ".", 
                "' ": "'", 
                " ,": ",", 
                "_": " "
            }, 
            sentence
        )
