import re

class Document(dict):
    PUNCTUATION_REGEX = re.compile(r'([\\\|¦\[\]\{\}$&®@%=~«»…½¡£₤§^`´¨#+.+,:;!?()/<>\-\+\*_"“”‘’\x08\x80-\xff\'])')
    HYPEN_REGEX = re.compile(r'(\w+)(-{2,})(\w+)')

    def open(path, vocabulary=None):
        text = open(path, 'r', encoding='utf-8').read().strip()
        return Document(text, vocabulary)

    def __init__(self, data, label=None, vocabulary=None, verbose=False):
        self.vocabulary = vocabulary
        self.label = label
        self.punctuation = {}
        if isinstance(data, str):
            frequencies = self.tokenize(data)
        elif isinstance(data, dict):
            frequencies = data
        else:
            raise 'Not supported'
        super().__init__(frequencies)
    
    def count(self, token):
        return self[token]
    
    def tokenize(self, data):
        frequencies = {}
        tokens = re.split(r'\s+', data.lower())
        for token in tokens:
            if token == '':
                continue
            if token in self.vocabulary:
                if token not in self:
                    frequencies[token] = 0
                frequencies[token] += 1
            else:
                token_punctuated = re.sub(Document.PUNCTUATION_REGEX, r' \1 ', token)
                token_punctuated_hyphened = re.sub(Document.HYPEN_REGEX, r'\1 \2 \3', token_punctuated)
                subtokens = re.split(r'\s+', token_punctuated_hyphened.strip())
                for subtoken in subtokens:
                    if subtoken in self.vocabulary:
                        if subtoken not in self:
                            frequencies[subtoken] = 0
                        frequencies[subtoken] += 1
                    else:
                        if subtoken not in self.punctuation:
                            self.punctuation[subtoken] = 0
                        self.punctuation[subtoken] += 1
        return frequencies
    
    def summary(self):
        print({ 'Corpus' : { 'known_tokens': self.known_tokens, 'unknown_tokens': self.unknown_tokens } })
