from src.utils.generators import tokenize

class BagOfWords(dict):
    def __init__(self, text_or_frequencies, vocabulary=None, expansions={}, replacements={}, ignored={}, **kwargs):
        # text_or_frequencies = list(text_or_frequencies)
        # print(text_or_frequencies)
        if isinstance(text_or_frequencies, dict):
            super().__init__(text_or_frequencies)
        else:
            tokens = tokenize(text_or_frequencies, expansions=expansions, vocabulary=vocabulary, replacements=replacements, ignored=ignored, **kwargs)
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
