import glob
import os
import re

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


class Corpus:
    def __init__(self, path_pattern, vocabulary, labeler):
        self.vocabulary = vocabulary
        self.labeler = labeler
        self.documents = []
        self.size = len(self.documents)
        self.labels = []
        self.known_tokens = set()
        self.unknown_tokens  = set()
        self.token_count = [0] * vocabulary.size
        self.load(path_pattern)

    def load(self, path_pattern, verbose=True):
        for path in glob.glob(path_pattern):
            label = path.split('/')[-2]
            self.labels.append(label)
            new_document = Document.open(path, self.vocabulary)
            self.documents.append(new_document)
            self.known_tokens = self.known_tokens.union(new_document.known_tokens)
            self.unknown_tokens = self.unknown_tokens.union(new_document.unknown_tokens)
            for token, count in new_document.token_count.items():
                if token not in self.vocabulary:
                    continue
                token_id = self.vocabulary.encode(token)
                self.token_count[token_id] += count
            if verbose and len(self.documents) % 1000 == 0:
                print('load:', len(self.documents), 'loaded')
        self.size = len(self.documents)


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
        self.priors = [None] * self.labeler.size
        self.count_tokens_with_label = [0] * self.labeler.size
        self.likelihoods = [ [None] * vocabulary.size for _ in self.labeler ]
        self.count_token_with_label = [ [0] * vocabulary.size for _ in self.labeler ]
    
    def fit(self, documents, labels):
        for document, label in zip(documents, labels):
            label_index = self.labeler.encode(label)
            self.count_tokens_with_label[label_index] += 1
            for token in document:
                if token in self.vocabulary:
                    token_index = self.vocabulary.encode(token)
                    token_count = document.count(token)
                    self.count_token_with_label[label_index][token_index] += token_count

    def prior(self, label):
        label_index = self.labeler.encode(label)
        prior = self.priors[label_index]
        if prior is None:
            prior = 1. * self.count_tokens_with_label[label_index] / sum(self.count_tokens_with_label)
            self.priors[label_index] = prior
        return prior

    def likelihood(self, token, label):
        if token not in self.vocabulary:
            return 0
        token_index = self.vocabulary.encode(token)
        label_index = self.labeler.encode(label)
        likelihood = self.likelihoods[label_index][token_index]
        if likelihood is None:
            likelihood = (self.count_token_with_label[label_index][token_index] + 1) / (self.count_tokens_with_label[label_index] + self.vocabulary.size)
            self.likelihoods[label_index][token_index] = likelihood
        return likelihood
    
    def posterior(self, label, document):
        posterior = self.prior(label)
        for token in document:
            posterior *= self.likelihood(token, label) ** document.count(token)
        return posterior
    
    def argmax(self, document, verbose=True):
        posteriors = [ self.posterior(label, document) for label in self.labeler ]
        if verbose:
            print('posteriors:', posteriors)
        return Utils.argmax(posteriors)

    def predict(self, document_or_documents):
        if isinstance(document_or_documents, Document):
            return self.argmax(document_or_documents)
        return [ self.argmax(document) for document in document_or_documents ]
