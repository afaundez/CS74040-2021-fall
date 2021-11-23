import re
from src.structures.bag_of_words import BagOfWords

PUNCTUATION_REGEX = re.compile(r'([\\\|¦\[\]\{\}$&®@%=~«»…½¡£₤§^`´¨#+.+,:;!?()/<>\-\+\*_"“”‘’\x08\x80-\xff\'])')
HYPEN_REGEX = re.compile(r'(\w+)(-{2,})(\w+)')

class Document(dict):
    def __init__(self, data, label=None, vocabulary=[], expansions={}, replacements={}, ignored=[], source=None, **kwargs):
        self.label = label
        self.source = source
        if isinstance(data, str):
            frequencies = BagOfWords(data, vocabulary=vocabulary, expansions=expansions, replacements=replacements, ignored=ignored, **kwargs)
        elif isinstance(data, dict):
            frequencies = data
        else:
            raise 'Not supported'
        super().__init__(frequencies)

    def open(path, *args, **kwargs):
        text = open(path).read().strip()
        return Document(text, *args, **kwargs)
    
    def count(self, token):
        return self[token]
