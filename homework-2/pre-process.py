import csv
from src.encoder import Encoder
from src.structures.corpus import Corpus

if __name__ == '__main__':
    labeler = Encoder(['action', 'comedy'])
    print(labeler)

    vocabulary = Encoder.open('movie-review-small/aclImdb/imdb.vocab')
    print(vocabulary)

    train_corpus = Corpus.open('movie-review-small/aclImdb/train/**/*.txt', vocabulary=vocabulary, verbose=True)
    print(train_corpus)
    train_corpus.write('movie-review-small/aclImdb/train.NB')

    test_corpus = Corpus.open('movie-review-small/aclImdb/test/**/*.txt', vocabulary=vocabulary, verbose=True)
    print(test_corpus)
    test_corpus.write('movie-review-small/aclImdb/test.NB')

    labeler = Encoder(['pos', 'neg'])
    print(labeler)

    with open('src/data/acronyms.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        acronyms = { acronym: meaning for acronym, meaning in reader }

    with open('src/data/smileys.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        smileys = { smiley: f'{labeler.decode(int(bias))}' for smiley, bias in reader }

    with open('src/data/positive-words.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        positive_words = { positive_word: f'{labeler.decode(int(bias))}' for positive_word, bias in reader }

    with open('src/data/negative-words.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        negative_words = { negative_word: f'{labeler.decode(int(bias))}' for negative_word, bias in reader }

    with open('src/data/negation.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        negations = { negation: token for negation, token in reader }

    with open('src/data/stopwords.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        stopwords = [ stopword for stopword in reader ]

    vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab', include=['||pos||', '||neg||', '||url||', '||email||', '||not||'])
    print(vocabulary)


    train_corpus = Corpus.open('movie-review-HW2/aclImdb/train/**/*.txt',
        vocabulary=vocabulary,
        replacements= {**smileys, **negative_words, **positive_words, **negations },
        expansions=acronyms,
        ignored=stopwords,
        verbose=True
    )
    print(train_corpus)
    train_corpus.frequencies.write('movie-review-HW2/aclImdb/train.vocab')
    train_corpus.write('movie-review-HW2/aclImdb/train.NB')

    test_corpus = Corpus.open('movie-review-HW2/aclImdb/test/**/*.txt',
        vocabulary=vocabulary,
        replacements= {**smileys, **negative_words, **positive_words, **negations },
        expansions=acronyms,
        ignored=stopwords,
        verbose=True
    )
    print(test_corpus)
    test_corpus.write('movie-review-HW2/aclImdb/test.NB', verbose=True)
