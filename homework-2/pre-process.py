from src.encoder import Encoder
from src.corpus import Corpus

if __name__ == '__main__':
    labeler = Encoder(['action', 'comedy'])
    print(labeler)

    vocabulary = Encoder.open('movie-review-small/aclImdb/imdb.vocab')
    print(vocabulary)

    train_corpus = Corpus.open('movie-review-small/aclImdb/train/**/*.txt', vocabulary, verbose=True)
    train_corpus.summary()
    train_corpus.write('movie-review-small/aclImdb/train.NB')

    test_corpus = Corpus.open('movie-review-small/aclImdb/test/**/*.txt', vocabulary, verbose=True)
    test_corpus.summary()
    test_corpus.write('movie-review-small/aclImdb/test.NB')

    labeler = Encoder(['post', 'neg'])
    labeler.summary()

    vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab')
    vocabulary.summary()

    train_corpus = Corpus.open('movie-review-HW2/aclImdb/train/**/*.txt', vocabulary, verbose=True)
    train_corpus.summary()
    train_corpus.write('movie-review-HW2/aclImdb/train.NB')

    test_corpus = Corpus.open('movie-review-HW2/aclImdb/test/**/*.txt', vocabulary, verbose=True)
    test_corpus.summary()
    test_corpus.write('movie-review-HW2/aclImdb/test.NB')
