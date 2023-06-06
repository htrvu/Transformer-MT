from torchtext.data.metrics import bleu_score
from torchtext.legacy.data import Field
import torch.nn as nn

from utils import translate_sentence
from typing import List

def bleu(
    valid_src: List[str],
    valid_trg: List[str],
    model: nn.Module,
    src_field: Field,
    trg_field: Field,
    device: str,
    k: int,
    max_len: int,
) -> float:
    """Calculate BLEU score

    Args:
        - valid_src (List[str]): Valid source sentences
        - valid_trg (List[str]): Valid target sentences
        - model (nn.Module): Model to be evaluated
        - src_field (Field): Tokenizer for source language
        - trg_field (Field): Tokenizer for target language
        - device (str): Device to run model (e.g. cuda:0, cpu)
        - k (int): Number of beams
        - max_len (int): Maximum length of sentence

    Returns: float: BLEU score
    """
    pred_sentences = []
    for sentence in valid_src:
        pred_trg = translate_sentence(
            sentence, model, src_field, trg_field, device, k, max_len
        )
        pred_sentences.append(pred_trg)

    pred_sentences = [trg_field.preprocess(sentence) for sentence in pred_sentences]
    trg_sentences = [[sent.split()] for sent in valid_trg]

    return bleu_score(pred_sentences, trg_sentences)
