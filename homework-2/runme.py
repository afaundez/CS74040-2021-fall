from src.encoder import Encoder
from src.corpus import Corpus
from src.document import Document
from src.model import Model
from src.metrics import Metrics

if __name__ == '__main__':
    labeler = Encoder(['action', 'comedy'])
    labeler.summary()

    vocabulary = Encoder.open('movie-review-small/aclImdb/imdb.vocab')
    vocabulary.summary()

    train_corpus = Corpus('movie-review-small/aclImdb/train/**/*.txt', vocabulary, labeler, verbose=True)
    train_corpus.summary()

    model = Model(vocabulary, labeler)
    model.fit(train_corpus, train_corpus.labels)
    model.summary()

    # document = Document.parse(' '.join(vocabulary), vocabulary)
    # prediction = model.predict(document, verbose=True)
    # print(f'predict({document})', prediction, labeler.decode(prediction))

    # document = 'couple,fly,fast'
    # prediction = model.predict(document)
    # print(f'predict({document})', prediction, labeler.decode(prediction))

    # print('model.priors', model.priors)
    # print('model.likelihoods', model.likelihoods)

    test_corpus = Corpus('movie-review-small/aclImdb/test/**/*.txt', vocabulary, labeler, verbose=True)
    test_corpus.summary()

    predictions = model.predict(test_corpus, verbose=True)
    print(f'predict({test_corpus})', predictions, labeler.decode(predictions))

    model.summary()

    Metrics.score(test_corpus.labels, labeler.decode(predictions))

    labeler = Encoder(['pos', 'neg'])
    labeler.summary()

    vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab')
    train_corpus = Corpus('movie-review-HW2/aclImdb/train/**/*.txt', vocabulary, labeler, verbose=True)
    model = Model(vocabulary, labeler)
    model.fit(train_corpus, train_corpus.labels)

    document = Document.open('movie-review-HW2/aclImdb/test/pos/0_10.txt', vocabulary)
    prediction = model.predict(document, verbose=True)
    print(prediction, labeler.decode(prediction))

    document = Document.open('movie-review-HW2/aclImdb/test/neg/0_2.txt', vocabulary)
    prediction = model.predict(document, verbose=True)
    print(prediction, labeler.decode(prediction))

    test_corpus = Corpus('movie-review-HW2/aclImdb/test/**/*.txt', vocabulary, labeler, verbose=True)
    print('test_corpus.size', test_corpus.size)

    predictions = model.predict(test_corpus)

    accuracy = Metrics.score(test_corpus.labels, labeler.decode(predictions))
    print('Metrics.score', accuracy)
