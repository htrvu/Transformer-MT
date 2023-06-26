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



def preprocess_text(input: str) -> str:
    '''
    Preprocess input sentence to feed into model

    Args:
        - input (str): input sentence

    Returns: (str) input sentence
    '''
    def check_mrs(input, i):
        is_mr = (i >= 2 and 
                input[i-2:i].lower() in ['mr', 'ms'] and
                (i < 3 or input[i-3] == ' '))
        is_mrs = (i >= 3 and 
                input[i-3:i].lower() == 'mrs' and 
                (i < 4 or input[i-4] == ' '))
        return is_mr or is_mrs

    def check_ABB_mid(content, i):
        if i <= 0:
            return False
        if i >= len(content)-1:
            return False
        l, r = content[i-1], content[i+1]
        return l.isupper() and r.isupper()

    def check_ABB_end(content, i):
        if i <= 0:
            return False
        l = content[i-1]
        return l.isupper()


    input += '.'

    # First step: replace special characters 
    check_list = ['\uFE16', '\uFE15', '\u0027','\u2018', '\u2019',
                    '“', '”', '\u3164', '\u1160', 
                    '\u0022', '\u201c', '\u201d', '"',
                    '[', '\ufe47', '(', '\u208d',
                    ']', '\ufe48', ')' , '\u208e', 
                    '—', '_', '–', '&']
    alter_chars = ['?', '!', '&apos;', '&apos;', '&apos;',
                    '&quot;', '&quot;', '&quot;', '&quot;', 
                    '&quot;', '&quot;', '&quot;', '&quot;', 
                    '&#91;', '&#91;', '&#91;', '&#91;',
                    '&#93;', '&#93;', '&#93;', '&#93;', 
                    '-', '-', '-', '&amp;']
    replace_dict = dict(zip(check_list, alter_chars))

    new_input = ''
    for i, char in enumerate(input):
        if char == '&' and (input[i:i+5] == '&amp;' or
                            input[i:i+6] == '&quot;' or
                            input[i:i+6] == '&apos;' or
                            input[i:i+5] == '&#93;' or
                            input[i:i+5] == '&#91;'):
            new_input += char
            continue
        new_input += replace_dict.get(char, char)
    input = new_input

    # Second step: add spaces
    check_sp_list = [',', '?', '!', '&apos;', '&amp;', '&quot;', '&#91;', 
                    '&#93;', '-', '/', '%', ':', '$', '#', '&', '*', ';', '=', '+', '@', '~', '>', '<']

    new_input = ''
    i = 0
    while i < len(input):
        char = input[i]
        found = False
        for string in check_sp_list:
            if string == input[i: i+len(string)]:
                new_input += ' ' + string 
                if string != '&apos;':
                    new_input += ' '
                i += len(string)
                found = True
                break
        if not found:
            new_input += char
            i += 1
    input = new_input

    new_input = ''
    for i, char in enumerate(input):
        if char != '.':
            new_input += char
            continue    
        elif check_mrs(input, i):
            # case 1: Mr. Mrs. Ms.
            new_input += '. '
        elif check_ABB_mid(input, i):
            # case 2: U[.]S.A.
            new_input += '.'
        elif check_ABB_end(input, i):
            # case 3: U.S.A[.]
            new_input += '. '
        else:
            new_input += ' . '

    input = new_input
    
    # Thrid step: remove not necessary spaces.
    new_input = ''
    for char in input:
        if new_input and new_input[-1] == ' ' and char == ' ':
            continue
        new_input += char
    input = new_input

    return input

def postprocess_text(output: str):
    '''
    Post process output sentence from model to display.

    Args:
        - output (str): output sentence from model

    Returns: (str) output sentence
    '''
    to_text = output
    to_text = re.sub('\s+', ' ', to_text)

    to_text = to_text.replace('<EOS>', '').replace('<pad>', '')
    to_text = to_text.replace('& quot ', '"')
    to_text = to_text.replace(' & quot', '"')
    to_text = to_text.replace('& apos ', "'")
    to_text = to_text.replace(' & apos', "'")
    to_text = to_text.replace('& # 91 ', "(")
    to_text = to_text.replace(' & # 93', ")")
    to_text = to_text.split('\\')[0].strip()

    return to_text