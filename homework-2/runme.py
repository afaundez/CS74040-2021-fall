from NB import Encoder, Document, Corpus, NB

class Metrics:
    def score(true_labels, pred_labels):
        tp = 0
        fp = 0
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label == pred_label:
                tp += 1
            else:
                fp += 1
        accuracy = tp / (tp + fp)
        print({ 'Metrics.score': { 'accuracy': accuracy } })

if __name__ == '__main__':
    labeler = Encoder(['action', 'comedy'])
    labeler.summary()

    vocabulary = Encoder.open('movie-review-small/aclImdb/imdb.vocab')
    vocabulary.summary()

    train_corpus = Corpus('movie-review-small/aclImdb/train/**/*.txt', vocabulary, labeler, verbose=True)
    train_corpus.summary()

    model = NB(vocabulary, labeler)
    model.fit(train_corpus, train_corpus.labels)
    model.summary()

    # document = Document.parse('furious,couple', vocabulary)
    # prediction = model.predict(document)
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
    model = NB(vocabulary, labeler)
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