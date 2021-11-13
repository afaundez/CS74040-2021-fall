import glob
import os
import re

class Vector(list):
    def __init__(self, dimensions=None, data=None, default=None, metric='value', name=''):
        self.dimensions = dimensions
        self.name = name
        self.metric = metric
        if data:
            super().__init__(data)
        else:
            super().__init__([default] * len(dimensions))
    
    def max_label_length(self):
        return max(len(str(label)) for label in [self.name, *self.dimensions])
    
    def max_value_cell_length(self):
        return max(len(str(value)) for value in [self.metric, *self])
    
    def __str__(self, default='', filler='-', joint='-+-', pad=' ', separator=' | '):
        header = separator.join([
            str(self.name).ljust(self.max_label_length(), pad),
            str(self.metric).rjust(self.max_value_cell_length(), pad)
        ])
        hline = joint.join([
            str(default).ljust(self.max_label_length(), filler),
            str(default).rjust(self.max_value_cell_length(), filler)
        ])
        rows = [
            separator.join([
                str(label).ljust(self.max_label_length(), pad),
                str(value).rjust(self.max_value_cell_length(), pad)
            ])
            for label, value in zip(self.dimensions, self)
        ]
        return '\n'.join([header, hline, *rows])

class Matrix(list):
    def __init__(self, rows, cols, data=None, default=None, name=''):
        self.rows = rows
        self.cols = cols
        self.name = name
        if data is None:
            data = [ [default] * len(self.cols) for _ in self.rows]
        self.extend(data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return super().__getitem__(row).__getitem__(col)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            super().__getitem__(row).__setitem__(col, value)
        else:
            super().__setitem__(key, value)
    
    def col_max_cell_lenght(self):
        max_cell_lenghts = [ len(col) for col in self.cols ]
        for row in self:
            for col, cell in enumerate(row):
                max_cell_lenghts[col] = max(max_cell_lenghts[col], len(str(cell)))
        return max_cell_lenghts
    
    def row_label_max_lenght(self):
        return max([ len(str(cell)) for cell in [self.name, *self.rows] ])

    
    def __str__(self, default='', filler='-', joint='-+-', pad=' ', separator=' | '):
        cell_lengths = [self.row_label_max_lenght()] + self.col_max_cell_lenght()
        header = separator.join([
            str(cell).rjust(lenght, pad)
            for cell, lenght in zip([self.name, *self.cols], cell_lengths)
        ])
        hline = joint.join([
            default.rjust(length, filler) for length in cell_lengths
        ])
        rows = [
            separator.join([
                str(cell).rjust(length, pad)
                for cell, length in zip([row_label, *self[row]], cell_lengths)
            ])
            for row, row_label in enumerate(self.rows)
        ]
        return '\n'.join([header, hline, *rows])

class Encoder(set):
    def open(path):
        with open(path, mode='r', encoding='utf-8') as file:
            tokens = [ line.strip() for line in file ]
            return Encoder(tokens)
    
    def __init__(self, values=[]):
        super().__init__(values)
        self.indexes = { value: index for index, value in enumerate(self) }
        self.values = { index: value for value, index in self.indexes.items() }
        self.size = len(self)

    def encode(self, value):
        return self.indexes[value]
    
    def decode(self, index_or_indexes):
        if isinstance(index_or_indexes, int):
            return self.values[index_or_indexes] 
        return [ self.values[index] for index in index_or_indexes ]
    
    def summary(self, size=10):
        print({ 'Encoder' : { 'size': self.size, 'indexes': self.indexes } })


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


class Utils:
    def argmax(iterable):
        max_index = None
        max_value = None
        for index, value in enumerate(iterable):
            if max_value is None or value > max_value:
                max_index = index
                max_value = value
        return max_index


class NB:
    def __init__(self, vocabulary, labeler):
        self.vocabulary = vocabulary
        self.labeler = labeler
        self.events_by_label = Vector(self.labeler, default=0, name='c(C)')
        self.events_by_label_by_token = Matrix(self.labeler, self.vocabulary, default=0, name='c(t,C)')
        self.priors = Vector(self.labeler, name='p(C)')
        self.likelihoods = Matrix(self.labeler, self.vocabulary, name='P(t|C)')
        self.posteriors = Matrix(self.vocabulary, self.labeler, name='P(C|t)')
    
    def fit(self, documents, labels):
        for document, label in zip(documents, labels):
            label_index = self.labeler.encode(label)
            self.events_by_label[label_index] += 1
            for token in document:
                if token in self.vocabulary:
                    token_index = self.vocabulary.encode(token)
                    token_count = document.count(token)
                    self.events_by_label_by_token[label_index, token_index] += token_count

    def prior(self, label):
        label_index = self.labeler.encode(label)
        prior = self.priors[label_index]
        if prior is None:
            prior = 1. * self.events_by_label[label_index] / sum(self.events_by_label)
            self.priors[label_index] = prior
        return prior

    def likelihood(self, token, label):
        if token not in self.vocabulary:
            return 0
        token_index = self.vocabulary.encode(token)
        label_index = self.labeler.encode(label)
        likelihood = self.likelihoods[label_index, token_index]
        if likelihood is None:
            likelihood = (self.events_by_label_by_token[label_index, token_index] + 1) / (sum(self.events_by_label_by_token[label_index]) + self.vocabulary.size)
            self.likelihoods[label_index, token_index] = likelihood
        return likelihood
    
    def posterior(self, label, document):
        posterior = self.prior(label)
        for token in document:
            posterior *= self.likelihood(token, label) ** document.count(token)
        return posterior
    
    def argmax(self, document, verbose=False):
        posteriors = Vector(self.labeler, data=[ self.posterior(label, document) for label in self.labeler ], name='p(C|d)')
        if verbose:
            print(posteriors)
        return Utils.argmax(posteriors)

    def predict(self, document_or_corpus, verbose=False):
        if isinstance(document_or_corpus, str):
            document = Document.parse(document_or_corpus, self.vocabulary)
            return self.argmax(document, verbose)
        elif isinstance(document_or_corpus, Document):
            return self.argmax(document_or_corpus, verbose)
        return [ self.argmax(document, verbose) for document in document_or_corpus ]
    
    def summary(self, size=10):
        print(self.events_by_label)
        print(self.priors)
        print(self.events_by_label_by_token)
        print(self.likelihoods)
