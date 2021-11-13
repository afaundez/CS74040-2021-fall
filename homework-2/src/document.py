import re

class Document(list):
    PUNCTUATION_REGEX = re.compile(r'([\\\|¦\[\]\{\}$&®@%=~«»…½¡£₤§^`´¨#+.+,:;!?()/<>\-\+\*_"“”‘’\x08\x80-\xff\'])')
    HYPEN_REGEX = re.compile(r'(\w+)(-{2,})(\w+)')

    def open(path, vocabulary):
        text = open(path, 'r', encoding='utf-8').read().strip()
        return Document(text, vocabulary)

    def parse(text, vocabulary):
        text = text
        return Document(text, vocabulary)

    def __init__(self, text, vocabulary):
        self.text = text
        self.vocabulary = vocabulary
        self.known_tokens = set()
        self.unknown_tokens = set()
        self.token_count = {}
        self.tokenize()
        super().__init__(self.known_tokens)
    
    def count(self, token):
        return self.token_count[token]
    
    def tokenize(self):
        tokens = re.split(r'\s+', self.text.lower())
        for token in tokens:
            if token != '' and token in self.vocabulary:
                if token not in self.token_count:
                    self.token_count[token] = 0
                    self.known_tokens.add(token)
                self.token_count[token] += 1
            else:
                token_punctuated = re.sub(Document.PUNCTUATION_REGEX, r' \1 ', token)
                token_punctuated_hyphened = re.sub(Document.HYPEN_REGEX, r'\1 \2 \3', token_punctuated)
                subtokens = re.split(r'\s+', token_punctuated_hyphened.strip())
                for subtoken in subtokens:
                    if subtoken != '' and subtoken in self.vocabulary:
                        if subtoken not in self.token_count:
                            self.token_count[subtoken] = 0
                            self.known_tokens.add(subtoken)
                        self.token_count[subtoken] += 1
                    else:
                        if subtoken not in self.unknown_tokens:
                            self.token_count[subtoken] = 0
                            self.unknown_tokens.add(subtoken)
                        self.token_count[subtoken] += 1
    
    def summary(self, size=10):
        print({ 'Corpus' : { 'known_tokens': self.known_tokens, 'unknown_tokens': self.unknown_tokens } })
