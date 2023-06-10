from torchtext.data.metrics import bleu_score
from torchtext.legacy.data import Field
import torch.nn as nn

from typing import List, Callable

from tqdm import tqdm

def bleu(
    valid_src: List[str],
    valid_trg: List[str],
    trg_field: Field,
    k: int,
    callback: Callable
) -> float:
    """Calculate BLEU score

    Args:
        - valid_src (List[str]): Valid source sentences
        - valid_trg (List[str]): Valid target sentences
        - trg_field (Field): Tokenizer for target language
        - k (int): Number of beams
        - callback (Callable): Callback function to be called after each epoch. 

    Returns: float: BLEU score
    """
    pred_sentences = []
    for sentence in tqdm(valid_src):
        pred_trg = callback(sentence, k)
        pred_sentences.append(pred_trg)

    pred_sentences = [trg_field.preprocess(sentence) for sentence in pred_sentences]
    trg_sentences = [[sent.split()] for sent in valid_trg]

    return bleu_score(pred_sentences, trg_sentences)
