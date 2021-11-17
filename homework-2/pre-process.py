from optparse import OptionParser

from src.encoder import Encoder
from src.structures.corpus import Corpus
from src.model import Model
from src.metrics import Metrics
from src.data import text_helpers

# python3 pre-process.py \
#     --labels pos,neg \
#     --input-train='movie-review-HW2/aclImdb/train/**/*.txt' \
#     --input-test='movie-review-HW2/aclImdb/test/**/*.txt' \
#     --output-path=movie-review-HW2/aclImdb \
#     --pre-process=acronym,smileys,positive-words,negative-words,negations,stopwords \
#     --ngrams=1 \
#     --use-train-vocabulary=True \
#     --use-imdb-vocab=True \
#     --extra-vocabulary=''

def parse_options():
    parser = OptionParser()
    parser.add_option("--labels", dest="labels", default='pos,neg')
    parser.add_option("--pre-process", dest="pre_process", default='acronym,smileys,positive-words,negative-words,negations,stopwords')
    parser.add_option("--ngrams", dest="ngrams", default='1')
    parser.add_option("--input-train", dest="input_train", default='movie-review-HW2/aclImdb/train/**/*.txt')
    parser.add_option("--input-test", dest="input_test", default='movie-review-HW2/aclImdb/test/**/*.txt')
    parser.add_option("--output-path", dest="output_path", default='movie-review-HW2/aclImdb')
    parser.add_option("--use-train-vocabulary", dest="use_train_vocabulary", default='True')
    parser.add_option("--use-imdb-vocab", dest="use_imdb_vocab", default='True')
    parser.add_option("--extra-vocabulary", dest="extra_vocabulary", default='')
    (options, args) = parser.parse_args()
    options = {
        'labels': options.labels.split(','),
        'pre-process': options.pre_process.split(','),
        'ngrams': int(options.ngrams),
        'input-train': options.input_train,
        'input-test': options.input_test,
        'output-path': options.output_path,
        'use-train-vocabulary': options.use_train_vocabulary == 'True',
        'use-imdb-vocab': options.use_imdb_vocab == 'True',
        'extra-vocabulary': options.extra_vocabulary.split(',')
    }
    options['results_path'] = f'{options["input-test"]}.output'
    options['output-prefix'] = '-'.join([*sorted(options['pre-process']), f'{options["ngrams"]}grams'])
    options['train-corpus-output'] = f'{options["output-path"]}/train-{options["output-prefix"]}{ "-tv" if options["use-train-vocabulary"] else "" }.NB'
    options['test-corpus-output'] = f'{options["output-path"]}/test-{options["output-prefix"]}{ "-tv" if options["use-train-vocabulary"] else "" }.NB'
    options['bigram'] = True if options['ngrams'] > 1 or options['use-train-vocabulary'] else False

    return options


def main():
    options = parse_options()

    if options['use-imdb-vocab']:
        include = options['extra-vocabulary']
        vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab', include=include)
        print(vocabulary)
    else:
        vocabulary = None

    labeler = Encoder(options['labels'])
    print(labeler)

    acronyms, smileys, positive_words, negative_words, negations, stopwords = text_helpers(labeler)

    expansions = {}
    if 'acronym' in options['pre-process']:
        expansions = { **expansions, **acronyms }

    replacements = {}
    if 'smileys' in options['pre-process']:
        replacements = { **replacements, **smileys}
    if 'positive-words' in options['pre-process']:
        replacements = { **replacements, **positive_words }
    if 'negative-words' in options['pre-process']:
        replacements = { **replacements, **negative_words }
    if 'negations' in options['pre-process']:
        replacements = { **replacements, **negations }

    ignored = []
    if 'stopwords' in options['pre-process']:
        ignored = [*ignored, *stopwords]

    train_corpus = Corpus.open(options['input-train'],
        bigrams=options['bigram'],
        vocabulary=vocabulary,
        replacements=replacements,
        expansions=expansions,
        ignored=ignored,
        verbose=True
    )
    print(train_corpus)
    train_corpus.write(options['train-corpus-output'])

    test_corpus = Corpus.open(options['input-test'],
        bigrams=options['bigram'],
        vocabulary=vocabulary,
        replacements=replacements,
        expansions=expansions,
        ignored=ignored,
        verbose=True
    )
    print(test_corpus)
    test_corpus.write(options['test-corpus-output'], verbose=True)

    return 0

if __name__ == '__main__':
    main()
