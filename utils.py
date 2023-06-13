import yaml
from typing import Dict
import re

def load_config(path: str) -> Dict:
    '''
    Load config from YAML file

    Args:
        - path (str): path to YAML file
    
    Returns: (dict) config
    '''
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def multiple_replace(dict: Dict, text: str) -> str:
    '''
    Replace multiple characters in a string

    Args:
        - dict (Dict): dictionary of characters to be replaced
        - text (str): string to be replaced

    Returns: replaced string
    '''
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)

