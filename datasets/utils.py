from typing import List, Tuple

def read_data(src_path: str, trg_file: str) -> Tuple[List[str], List[str]]:
    """
    Load source and target data from given files. For example: train.en (English) and train.fr (French)
    
    Args:
        src_path (str): path to source file
        trg_file (str): path to target file

    Returns: Tuple[List[str], List[str]]: source data and target data (readable text)
    """
    src_data = open(src_path, "r", encoding="utf8").read().strip().split("\n")
    trg_data = open(trg_file, "r", encoding="utf8").read().strip().split("\n")
    return src_data, trg_data