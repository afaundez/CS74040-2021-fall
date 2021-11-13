import math
from src.document import Document
from src.matrix import Matrix
from src.vector import Vector
from src.utils import Utils

class Model:
    def __init__(self, vocabulary, labeler, log=False):
        self.vocabulary = vocabulary
        self.labeler = labeler
        self.log = log
        self.events_by_label = Vector(self.labeler, default=0, name='c(C)')
        self.events_by_label_by_token = Matrix(self.labeler, self.vocabulary, default=0, name='c(t,C)')
        self.priors = Vector(self.labeler, name=Utils.loggify('p(C)', self.log))
        self.likelihoods = Matrix(self.labeler, self.vocabulary, name=Utils.loggify('P(t|C)', self.log))
    
    def fit(self, documents, labels, verbose=False):
        for index, (document, label) in enumerate(zip(documents, labels)):
            label_index = self.labeler.encode(label)
            self.events_by_label[label_index] += 1
            for token in document:
                if token in self.vocabulary:
                    token_index = self.vocabulary.encode(token)
                    token_count = document.count(token)
                    self.events_by_label_by_token[label_index, token_index] += token_count
            if verbose and index % 1000 == 0:
                print(index, 'documents fitted')

    def prior(self, label):
        label_index = self.labeler.encode(label)
        prior = self.priors[label_index]
        if prior is None:
            prior = 1. * self.events_by_label[label_index] / sum(self.events_by_label)
            self.priors[label_index] = math.log(prior, 2) if self.log else prior
        return prior

    def likelihood(self, token, label):
        if token not in self.vocabulary:
            return 0
        token_index = self.vocabulary.encode(token)
        label_index = self.labeler.encode(label)
        likelihood = self.likelihoods[label_index, token_index]
        if likelihood is None:
            likelihood = (self.events_by_label_by_token[label_index, token_index] + 1) / (sum(self.events_by_label_by_token[label_index]) + self.vocabulary.size)
            self.likelihoods[label_index, token_index] = math.log(likelihood, 2) if self.log else likelihood
        return likelihood
    
    def posterior(self, label, document):
        posterior = self.prior(label)
        for token in document:
            if self.log:
                posterior += document.count(token) * self.likelihood(token, label)
            else:
                posterior *= self.likelihood(token, label) ** document.count(token)
        return posterior
    
    def labelize(self, document, verbose=False):
        posteriors = Vector(self.labeler, data=[ self.posterior(label, document) for label in self.labeler ], name='p(C|d)')
        if verbose:
            print(posteriors)
        return Utils.argmax(posteriors)

    def predict(self, document_or_corpus, verbose=False):
        if isinstance(document_or_corpus, str):
            document = Document.parse(document_or_corpus, self.vocabulary)
            return self.labelize(document, verbose)
        elif isinstance(document_or_corpus, Document):
            return self.labelize(document_or_corpus, verbose)
        results = []
        for document in document_or_corpus:
            results.append(self.labelize(document))
            if verbose and len(results) % 1000 == 0:
                print(len(results), 'documents predicted')
        return results
    
    def __str__(self):
        return f'Model(vocabulary={self.vocabulary}, labeler={self.labeler})'
    
    def summary(self, size=10):
        print(self.events_by_label)
        print(self.priors)
        print(self.events_by_label_by_token)
        print(self.likelihoods)
