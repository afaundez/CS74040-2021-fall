from itertools import chain
from src.utils.generators import tokenize, pairwise

class BagOfWords(dict):
    def __init__(self, text_or_frequencies, bigrams=False, vocabulary=None, expansions={}, replacements={}, ignored={}, **kwargs):
        if isinstance(text_or_frequencies, dict):
            super().__init__(text_or_frequencies)
        else:
            tokens = tokenize(text_or_frequencies, expansions=expansions, vocabulary=vocabulary, replacements=replacements, ignored=ignored, **kwargs)

            if bigrams:
                tokens = map(lambda pair: ' '.join(pair), pairwise(chain(['<s>'], tokens, ['</s>'])))
            tokens = list(tokens)
            for token in tokens:
                if token not in self:
                    self[token] = 0
                self[token] += 1
    
    def open(path, *args, **kwargs):
        with open(path, 'r') as file:
            return BagOfWords(file.read(), *args, **kwargs)

    def merge(self, other):
        for token, count in other.items():
            if token not in self:
                self[token] = 0
            self[token] += count
        return self
    
    def write(self, filename):
        with open(filename, 'w') as f:
            for token, count in self.items():
                f.write(f'{token} {count}\n')
