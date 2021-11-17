from optparse import OptionParser

from src.encoder import Encoder
from src.structures.corpus import Corpus
from src.data import text_helpers

# python3 pre-process.py \
#     --training-file='movie-review-HW2/aclImdb/train/**/*.txt' \
#     --test-file='movie-review-HW2/aclImdb/test/**/*.txt' \
#     --output-path=movie-review-HW2/aclImdb \
#     --vocabulary-file=movie-review-HW2/aclImdb/imdb.vocab \
#     --add-label=pos --add-label=neg \
#     --ngrams=1

def parse_options():
    parser = OptionParser()
    parser.add_option("--training-file", dest="training_file", default='movie-review-HW2/aclImdb/train/**/*.txt')
    parser.add_option("--test-file", dest="test_file", default='movie-review-HW2/aclImdb/test/**/*.txt')

    parser.add_option("--output-path", dest="output_path", default='movie-review-HW2/aclImdb')
    parser.add_option("--output-training-file", dest="output_training_file", default='<output_path>/train-<ngram>gram-<pre-process-1>-...-<pre-process-1>.NB')
    parser.add_option("--output-test-file", dest="output_test_file", default='<output_path>/test-<ngram>gram-<pre-process-1>-...-<pre-process-1>.NB')

    parser.add_option("--vocabulary-file", dest="vocabulary_file", default='movie-review-HW2/aclImdb/imdb.vocab')

    
    parser.add_option("--add-label", action="append", dest="labels", default=[])
    parser.add_option("--ngrams", dest="ngrams", default='1', type=int)
    parser.add_option("--use-train-vocabulary", action="store_true", default=False, dest="use_train_vocabulary")

    parser.add_option("--add-pre-process", default=[], dest="pre_process", action="append", choices=('acronym', 'smileys', 'positive-words', 'negative-words', 'negations', 'stopwords'))
    parser.add_option("--add-token", default=[], dest="extra_vocabulary", action="append")

    (options, args) = parser.parse_args()

    output_ids = [f'{str(options.ngrams)}grams', *sorted(options.pre_process)]
    output_id = '-'.join([ value for value in output_ids if value ])
    options.output_id = output_id

    options.bigrams = True if options.ngrams > 1 else False # TODO: use options.ngrams
    
    if options.ngrams > 1:
        options.use_train_vocabulary = True

    if options.output_training_file == '<output_path>/train-<ngram>gram-<pre-process-1>-...-<pre-process-1>.NB':
        options.output_training_file = f'{options.output_path}/train-{options.output_id}.NB'

    print(options.output_id)
    if options.output_test_file == '<output_path>/test-<ngram>gram-<pre-process-1>-...-<pre-process-1>.NB':
        options.output_test_file = f'{options.output_path}/test-{options.output_id}.NB'

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

    labeler = Encoder(options.labels)
    print(labeler)

    acronyms, smileys, positive_words, negative_words, negations, stopwords = text_helpers(labeler)

    expansions = {}
    if 'acronym' in options.pre_process:
        expansions = { **expansions, **acronyms }

    replacements = {}
    if 'smileys' in options.pre_process:
        replacements = { **replacements, **smileys}
    if 'positive-words' in options.pre_process:
        replacements = { **replacements, **positive_words }
    if 'negative-words' in options.pre_process:
        replacements = { **replacements, **negative_words }
    if 'negations' in options.pre_process:
        replacements = { **replacements, **negations }

    ignored = []
    if 'stopwords' in options.pre_process:
        ignored = [*ignored, *stopwords]

    train_corpus = Corpus.open(options.training_file,
        bigrams=options.bigrams,
        vocabulary=vocabulary,
        replacements=replacements,
        expansions=expansions,
        ignored=ignored,
        verbose=True
    )
    print(train_corpus)
    train_corpus.write(options.output_training_file)

    test_corpus = Corpus.open(options.test_file,
        bigrams=options.bigrams,
        vocabulary=vocabulary,
        replacements=replacements,
        expansions=expansions,
        ignored=ignored,
        verbose=True
    )
    print(test_corpus)
    test_corpus.write(options.output_test_file, verbose=True)

    return 0

if __name__ == '__main__':
    main()
