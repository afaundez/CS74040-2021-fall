from optparse import OptionParser

from src.encoder import Encoder
from src.structures.corpus import Corpus
from src.model import Model
from src.metrics import Metrics

# python3 NB.py \
#     --training-file=movie-review-HW2/aclImdb/train-1grams.NB \
#     --test-file=movie-review-HW2/aclImdb/test-1grams.NB \
#     --output-file=movie-review-HW2/aclImdb/test-1grams.NB.output \
#     --vocabulary-file=movie-review-HW2/aclImdb/imdb.vocab \
#     --use-train-vocabulary

def parse_options():
    parser = OptionParser()
    parser.add_option("--training-file", dest="training_file", default='movie-review-HW2/aclImdb/train-1grams.NB')
    parser.add_option("--test-file", dest="test_file", default='movie-review-HW2/aclImdb/test-1grams.NB')
    parser.add_option("--output-file", dest="output_file", default='<test-file>.output')

    parser.add_option("--vocabulary-file", dest="vocabulary_file", default='movie-review-HW2/aclImdb/imdb.vocab')
    parser.add_option("--add-token", default=[], dest="extra_vocabulary", action="append")
    parser.add_option("--limit-predictions", dest="limit_predictions", default=None, type=int)
    
    parser.add_option("--use-train-vocabulary", default=False, action="store_true", dest="use_train_vocabulary")

    (options, args) = parser.parse_args()

    if options.output_file == '<test-file>.output':
        options.output_file = f'{options.test_file}.output'

    for option in options.__dict__:
        print(f'{option}: {options.__dict__[option]}')

    return options

def main():
    options = parse_options()

    if options.vocabulary_file:
        include = options.extra_vocabulary
        vocabulary = Encoder.open(options.vocabulary_file, include=include)
        print(vocabulary)
    else:
        vocabulary = None

    train_corpus = Corpus.open(options.training_file, frequencies=True, verbose=True)
    test_corpus = Corpus.open(options.test_file, frequencies=True, verbose=True)

    if not options.vocabulary_file or options.use_train_vocabulary:
        train_labels = list(train_corpus.frequencies.keys())
        vocabulary = Encoder(train_labels)

    train_labels = list(dict.fromkeys(train_corpus.labels()))
    labeler = Encoder(train_labels)

    model = Model(vocabulary, labeler, log=True)
    model.fit(train_corpus, train_corpus.labels(), verbose=True)

    predictions = model.predict(test_corpus.documents(limit=options.limit_predictions), verbose=True, debug=False)
    score = Metrics.score(test_corpus.labels(limit=options.limit_predictions), labeler.decode(predictions), labeler)
    print(score)

    with open(options.output_file, 'w') as file:
        file.write(f'{options}\n\n')
        file.write(f'{labeler}\n\n')
        file.write(f'{vocabulary}\n\n')
        file.write(f'{train_corpus}\n\n')
        file.write(f'{test_corpus}\n\n')
        file.write(f'{model}\n\n')
        print(score['confusion'], file=file)
        print('accuracy', score['accuracy'], file=file)

    return 0

if __name__ == '__main__':
    main()
