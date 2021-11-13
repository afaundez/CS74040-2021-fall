import glob
from src.document import Document

class Corpus(list):
    def __init__(self, path_pattern, vocabulary, labeler, verbose=False):
        self.vocabulary = vocabulary
        self.labeler = labeler
        self.labels = []
        self.known_tokens = set()
        self.unknown_tokens  = set()
        self.token_count = [0] * vocabulary.size
        super().__init__([])
        self.load(path_pattern, verbose=verbose)
        self.size = len(self)

    def load(self, path_pattern, verbose=False):
        for path in glob.glob(path_pattern):
            label = path.split('/')[-2]
            self.labels.append(label)
            new_document = Document.open(path, self.vocabulary)
            self.append(new_document)
            self.known_tokens = self.known_tokens.union(new_document.known_tokens)
            self.unknown_tokens = self.unknown_tokens.union(new_document.unknown_tokens)
            for token, count in new_document.token_count.items():
                if token not in self.vocabulary:
                    continue
                token_id = self.vocabulary.encode(token)
                self.token_count[token_id] += count
            if verbose and len(self) % 1000 == 0:
                print('Corpus.load:', len(self), 'loaded')
    
    def summary(self, size=10):
        print({ 'Corpus' : { 'known_tokens': self.known_tokens, 'unknown_tokens': self.unknown_tokens } })
