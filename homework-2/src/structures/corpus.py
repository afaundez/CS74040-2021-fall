import glob
import json

from src.structures.document import Document
from src.tokenized_text import BagOfWords
from src.utils.decorators import incremental

class Corpus(list):
    def __init__(self, document_or_documents, vocabulary=[], expansions=[], replacements=[], ignored=[], **kwargs):
        self.frequencies = BagOfWords({})
        for item in self.generate(document_or_documents, **kwargs):
            self.load(item, vocabulary=vocabulary, expansions=expansions, replacements=replacements, ignored=ignored, **kwargs)
    
    def append(self, document):
        self.frequencies.merge(document)
        return super().append(document)

    @incremental('documents loaded')
    def generate(self, document_or_documents, *args, **kwargs):
        if isinstance(document_or_documents, Document):
            yield document_or_documents
        elif isinstance(document_or_documents, str):
            yield document_or_documents
        else:
            yield from document_or_documents


    def labels(self):
        return [ document.label for document in self ]

    def load(self, item, label=None, **kwargs):
        if isinstance(item, Document):
            document = item
        elif isinstance(item, str):
            text, label = (item, label)
            document =  Document(text, label=label, **kwargs)
        elif isinstance(item, tuple):
            text, label = item
            document = Document(text, label=label, **kwargs)
        elif isinstance(item, dict):
            frequencies, label = (item['frequencies'], item['label'])
            document = Document(frequencies, label=label, **kwargs)
        self.append(document)
        return document
    
    def open(filename_or_pattern, frequencies=False, **kwargs):
        generator = Corpus.open_generator(filename_or_pattern, frequencies=frequencies)
        return Corpus(generator, **kwargs)

    def open_generator(filename_or_pattern, frequencies=False):
        filenames  = glob.glob(filename_or_pattern)
        for filename in filenames:
            if frequencies:
                with open(filename) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line.strip())
            else:
                label = filename.split('/')[-2]
                text = open(filename).read().strip()
                yield text, label

    def write(self, path, **kwargs):
        with open(path, 'w') as f:
            for document in self.write_iterator(**kwargs):
                f.write(json.dumps({ 'frequencies': document, 'label': document.label }) + '\n')
    
    @incremental('documents written')
    def write_iterator(self, **kwargs):
        yield from self
    
    def __str__(self):
        return f'Corpus(documents={len(self)}, tokens={len(self.frequencies)}, words={sum(self.frequencies.values())}))'
