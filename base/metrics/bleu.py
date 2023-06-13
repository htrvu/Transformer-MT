from torchtext.data.metrics import bleu_score
from typing import List

def calc_bleu(
        pred_sentences: List[str],
        trg_sentences: List[str],
        max_n_gram: int = 4
) -> float:
    """Calculate BLEU score between predicted and target sentences

    Args:
        - pred_sentences (List[str]): List of predicted sentences
        - trg_sentences (List[str]): List of target sentences
        - max_n_gram (int): Maximum n-gram to be used in BLEU score calculation

    Returns: float: BLEU score
    """
    return bleu_score(pred_sentences, trg_sentences, max_n_gram)
