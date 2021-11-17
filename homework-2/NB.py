from optparse import OptionParser

from src.encoder import Encoder
from src.structures.corpus import Corpus
from src.model import Model
from src.metrics import Metrics
from src.data import text_helpers

# python3 NB.py \
#     --input-train=movie-review-HW2/aclImdb/train-1grams.NB \
#     --input-test=movie-review-HW2/aclImdb/test-1grams.NB \
#     --use-train-vocabulary=True \
#     --use-imdb-vocab=True \
#     --extra-vocabulary=''

def parse_options():
    parser = OptionParser()
    parser.add_option("--input-train", dest="input_train", default='movie-review-HW2/aclImdb/train-1grams.NB')
    parser.add_option("--input-test", dest="input_test", default='movie-review-HW2/aclImdb/test-1grams.NB')
    parser.add_option("--use-train-vocabulary", dest="use_train_vocabulary", default='True')
    parser.add_option("--use-imdb-vocab", dest="use_imdb_vocab", default='True')
    parser.add_option("--extra-vocabulary", dest="extra_vocabulary", default='')
    parser.add_option("--limit-predictions", dest="limit_predictions", default='')
    (options, args) = parser.parse_args()
    options = {
        'input-train': options.input_train,
        'input-test': options.input_test,
        'use-train-vocabulary': options.use_train_vocabulary == 'True',
        'use-imdb-vocab': options.use_imdb_vocab == 'True',
        'extra-vocabulary': options.extra_vocabulary.split(','),
        'limit-predictions': int(options.limit_predictions) if options.limit_predictions != '' else None
    }
    options['results_path'] = f'{options["input-test"]}.output'
    return options

def main():
    options = parse_options()

    if options['use-imdb-vocab']:
        include = options['extra-vocabulary']
        vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab', include=include)
        print(vocabulary)
    else:
        vocabulary = None

    train_corpus = Corpus.open(options['input-train'], frequencies=True, verbose=True)
    test_corpus = Corpus.open(options['input-test'], frequencies=True, verbose=True)

    if not options['use-imdb-vocab'] or options['use-train-vocabulary']:
        vocabulary = Encoder(list(train_corpus.frequencies.keys()))

    train_labels = list(dict.fromkeys(train_corpus.labels()))
    labeler = Encoder(train_labels)

    model = Model(vocabulary, labeler, log=True)
    model.fit(train_corpus, train_corpus.labels(), verbose=True)

    predictions = model.predict(test_corpus.documents(limit=options['limit-predictions']), verbose=True, debug=False)
    score = Metrics.score(test_corpus.labels(limit=options['limit-predictions']), labeler.decode(predictions), labeler)
    print(score)

    with open(options["results_path"], 'w') as file:
        file.write(f'{score}\n\n')
        file.write(f'{labeler}\n\n')
        file.write(f'{vocabulary}\n\n')
        file.write(f'{train_corpus}\n\n')
        file.write(f'{test_corpus}\n\n')
        file.write(f'{model}\n\n')
        print(score['confusion'], file=file)
        file.write(f'{options}\n\n')

    return 0

if __name__ == '__main__':
    main()
