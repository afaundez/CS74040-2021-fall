import math
from src.document import Document
from src.matrix import Matrix
from src.vector import Vector
from src.utils import Utils

class Model:
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
    
    def posterior(self, label, document, log=False):
        posterior = math.log(self.prior(label), 2)
        for token in document:
            posterior += math.log(self.likelihood(token, label), 2) * document.count(token) 
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
