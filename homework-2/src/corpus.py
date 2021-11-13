import glob
import json
from src.document import Document

class Corpus(list):
    def __init__(self, generator, vocabulary=None, verbose=False):
        self.vocabulary = vocabulary
        for item in generator:
            self.load(item, verbose=verbose)
            if verbose and len(self) % 1000 == 0:
                print(f'{len(self)} documents loaded')


    def labels(self):
        return [ document.label for document in self ]

    def load(self, item, label=None, verbose=False):
        if isinstance(item, Document):
            document = item
        elif isinstance(item, str):
            text, label = (item, label)
            document =  Document(text, label=label, vocabulary=self.vocabulary, verbose=verbose)
        elif isinstance(item, tuple):
            text, label = item
            document = Document(text, label=label, vocabulary=self.vocabulary, verbose=verbose)
        elif isinstance(item, dict):
            frequencies, label = (item['frequencies'], item['label'])
            document = Document(frequencies, label=label, vocabulary=self.vocabulary, verbose=verbose)
        self.append(document)
    
    def open(filename_or_pattern, vocabulary=None, frequencies=False, verbose=False):
        return Corpus(Corpus.open_generator(filename_or_pattern, frequencies=frequencies, verbose=verbose), vocabulary=vocabulary, verbose=verbose)

    
    def open_generator(filename_or_pattern, frequencies=False, verbose=False):
        filenames  = glob.glob(filename_or_pattern)
        for filename in filenames:
            if frequencies:
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield json.loads(line)
            else:
                label = filename.split('/')[-2]
                text = open(filename, 'r', encoding='utf-8').read().strip()
                yield text, label
    
    def summary(self):
        print({ 'Corpus' : { 'size': len(self) } })
    
    def __str__(self):
        return f'Corpus(size={len(self)}, vocabulary={self.vocabulary})'

    def write(self, path):
        with open(path, 'w') as f:
            for document in self:
                f.write(json.dumps({ 'frequencies': document, 'label': document.label }) + '\n')
