from src.encoder import Encoder
from src.corpus import Corpus
from src.document import Document
from src.model import Model
from src.metrics import Metrics

if __name__ == '__main__':
    labeler = Encoder(['action', 'comedy'])
    vocabulary = Encoder.open('movie-review-small/aclImdb/imdb.vocab')

    train_corpus = Corpus.open('movie-review-small/aclImdb/train.NB', vocabulary, frequencies=True, verbose=True)
    print(train_corpus)
    test_corpus = Corpus.open('movie-review-small/aclImdb/test.NB', vocabulary, frequencies=True, verbose=True) 
    print(test_corpus)

    model = Model(vocabulary, labeler)
    model.fit(train_corpus, train_corpus.labels())
    print(model)

    predictions = model.predict(test_corpus, verbose=True)
    print(f'predict({test_corpus})', predictions, labeler.decode(predictions))
    model.summary()

    score = Metrics.score(test_corpus.labels(), labeler.decode(predictions))
    print(score)

    labeler = Encoder(['pos', 'neg'])
    vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab')

    train_corpus = Corpus.open('movie-review-HW2/aclImdb/train.NB', vocabulary, frequencies=True, verbose=True)
    print(train_corpus)
    test_corpus = Corpus.open('movie-review-HW2/aclImdb/test.NB', vocabulary, frequencies=True, verbose=True) 
    print(test_corpus)

    model = Model(vocabulary, labeler, log=True)
    model.fit(train_corpus, train_corpus.labels(), verbose=True)
    print(model)

    predictions = model.predict(test_corpus, verbose=True)
    score = Metrics.score(test_corpus.labels(), labeler.decode(predictions))
    print(score)