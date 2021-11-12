from NB import Encoder, Document, Corpus, NB

if __name__ == '__main__':
    labeler = Encoder(['action', 'comedy'])
    print('labeler.indexes', labeler.indexes)
    print('labeler.values', labeler.values)

    vocabulary = Encoder.open('movie-review-small/aclImdb/imdb.vocab')
    print('vocabulary.indexes', vocabulary.indexes)

    train_corpus = Corpus('movie-review-small/aclImdb/train/**/*.txt', vocabulary, labeler)
    print('train_corpus.known_tokens', train_corpus.known_tokens)
    print('train_corpus.token_count', train_corpus.token_count)

    model = NB(vocabulary, labeler)
    model.fit(train_corpus.documents, train_corpus.labels)
    print('model.count_tokens_with_label', model.count_tokens_with_label)
    print('model.count_token_with_label', model.count_token_with_label)

    document = Document.parse('fast,couple,shoot,fly', vocabulary)
    prediction = model.predict(document)
    print(f'predict({document})', prediction, labeler.decode(prediction))

    print('model.priors', model.priors)
    print('model.likelihoods', model.likelihoods)

    document = Document.parse('furious,couple', vocabulary)
    prediction = model.predict(document)
    print(f'predict({document})', prediction, labeler.decode(prediction))

    document = Document.parse('couple,fly,fast', vocabulary)
    prediction = model.predict(document)
    print(f'predict({document})', prediction, labeler.decode(prediction))

    print('model.priors', model.priors)
    print('model.likelihoods', model.likelihoods)






    labeler = Encoder(['pos', 'neg'])
    print('labeler.indexes', labeler.indexes)
    print('labeler.values', labeler.values)

    vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab')
    train_corpus = Corpus('movie-review-HW2/aclImdb/train/**/*.txt', vocabulary, labeler)
    model = NB(vocabulary, labeler)
    model.fit(train_corpus.documents, train_corpus.labels)

    document = Document.open('movie-review-HW2/aclImdb/test/pos/0_10.txt', vocabulary)
    prediction = model.predict(document)
    print(f'predict({document})', prediction, labeler.decode(prediction))

    document = Document.open('movie-review-HW2/aclImdb/test/neg/0_2.txt', vocabulary)
    prediction = model.predict(document)
    print(f'predict({document})', prediction, labeler.decode(prediction))
