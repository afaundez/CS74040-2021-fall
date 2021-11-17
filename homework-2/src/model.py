import math
from src.structures.document import Document
from src.structures.matrix import Matrix
from src.structures.vector import Vector
from src.utils.processors import argmax, loggify
from src.utils.decorators import incremental

class Model:
    def __init__(self, vocabulary, labeler, log=False):
        self.vocabulary = vocabulary
        self.labeler = labeler
        self.log = log
        self.events_by_label = Vector(self.labeler, default=0, name='c(C)')
        self.events_by_label_by_token = Matrix(self.labeler, self.vocabulary, default=0, name='c(t,C)')
        self.priors = Vector(self.labeler, name=loggify('p(C)', self.log))
        self.likelihoods = Matrix(self.labeler, self.vocabulary, name=loggify('P(t|C)', self.log))
        self.events_by_label_by_token_sum =  Vector(self.labeler, default=None, name='sum')
        self.events_sum = None
    

    def fit(self, documents, labels, **kwargs):
        for document, label in self.fit_generator(documents, labels, **kwargs):
            label_index = self.labeler.encode(label)
            self.events_by_label[label_index] += 1
            for token in document:
                if token in self.vocabulary:
                    token_index = self.vocabulary.encode(token)
                    token_count = document.count(token)
                    self.events_by_label_by_token[label_index, token_index] += token_count
    
    @incremental('documents fitted')
    def fit_generator(self, documents=[], labels=[], **kwargs):
        yield from zip(documents, labels)

    def sum_events(self):
        tmp_sum_events = self.events_sum
        if tmp_sum_events is None:
            tmp_sum_events = sum(self.events_by_label)
            self.events_sum = tmp_sum_events
        return tmp_sum_events

    def prior(self, label):
        label_index = self.labeler.encode(label)
        prior = self.priors[label_index]
        if prior is None:
            prior = 1. * self.events_by_label[label_index] / self.sum_events()
            self.priors[label_index] = math.log(prior, 2) if self.log else prior
        return prior


    def sum_events_by_label_by_token(self, label):
        label_index = self.labeler.encode(label)
        tmp_sum_events_by_label_by_token = self.events_by_label_by_token_sum[label_index]
        if tmp_sum_events_by_label_by_token is None:
            tmp_sum_events_by_label_by_token = sum(self.events_by_label_by_token[label_index])
            self.events_by_label_by_token_sum[label_index] = tmp_sum_events_by_label_by_token
        return tmp_sum_events_by_label_by_token

    def likelihood(self, token, label):
        if token not in self.vocabulary:
            return 0
        token_index = self.vocabulary.encode(token)
        label_index = self.labeler.encode(label)
        likelihood = self.likelihoods[label_index, token_index]
        if likelihood is None:
            likelihood = (self.events_by_label_by_token[label_index, token_index] + 1) / (self.sum_events_by_label_by_token(label) + self.vocabulary.size)
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
    
    def labelize(self, document, verbose=False, debug=False):
        posteriors = Vector(self.labeler, data=[ self.posterior(label, document) for label in self.labeler ], name='p(C|d)')
        if debug:
            print(posteriors)
        return argmax(posteriors)

    def predict(self, document_or_corpus, **kwargs):
        if isinstance(document_or_corpus, str):
            document = Document.parse(document_or_corpus, self.vocabulary)
            return self.labelize(document, **kwargs)
        elif isinstance(document_or_corpus, Document):
            return self.labelize(document_or_corpus, **kwargs)
        else:
            return [item for item in self.generate_predictions(document_or_corpus, **kwargs)]
    
    @incremental('documents predicted')
    def generate_predictions(self, document_or_corpus, **kwargs):
        for document in document_or_corpus:
            yield self.labelize(document, **kwargs)
    
    def __str__(self):
        return f'Model(vocabulary={self.vocabulary}, labeler={self.labeler})'
    
    def summary(self, size=10):
        print(self.events_by_label)
        print(self.priors)
        print(self.events_by_label_by_token)
        print(self.likelihoods)
