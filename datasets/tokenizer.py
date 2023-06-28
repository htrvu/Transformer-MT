
import spacy
import re
from typing import List

class Tokenizer:
    def __init__(self, lang):
        """
        Args:
            - lang (str): language to be used for tokenization
        """
        self.nlp = spacy.load(lang)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy

        Args:
            - text (str): text to be tokenized

        Returns: List[str]: tokenized texts
        """
        # text = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(text))
        text = re.sub(r"[\*\n\\\+\/\=•\|]", " ", str(text))
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r"\:+", ":", text)
        text = re.sub(r"\;+", ";", text)
        text = re.sub(r"\?+", "?", text)
        text = text.lower()
        try:
            return [tok.text for tok in self.nlp.tokenizer(text) if tok.text != " "]
        except:
            # Strange behavior of spacy sometimes, text is blank after tokenization
            return []
