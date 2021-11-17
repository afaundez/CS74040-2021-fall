import csv
from src.encoder import Encoder
from src.structures.corpus import Corpus
from src.model import Model
from src.metrics import Metrics

options = {
    'labels': ['pos', 'neg'],
    'pre-process': [],
    'ngrams': 2,
    'train_input': 'movie-review-HW2/aclImdb/train/**/*.txt',
    'test_input': 'movie-review-HW2/aclImdb/test/**/*.txt',
    'output_path': 'movie-review-HW2/aclImdb',
    'use_train_vocabulary': True,
    'parse_corpus': True,
    'use_imdb_vocab': True,
    'include': [], #['||pos||', '||neg||', '||url||', '||email||', '||not||'],
    'limit_predictions': None,
}

options['output_prefix'] = '-'.join([*sorted(options['pre-process']), f'{options["ngrams"]}grams'])
options['train_corpus_output'] = f'{options["output_path"]}/train-{options["output_prefix"]}{ "-tv" if options["use_train_vocabulary"] else "" }.NB'
options['test_corpus_output'] = f'{options["output_path"]}/test-{options["output_prefix"]}{ "-tv" if options["use_train_vocabulary"] else "" }.NB'
options['bigram'] = True if options['ngrams'] > 1 else False
if options['bigram']:
    options['use_train_vocabulary'] = True
options['results_path'] = f'{options["output_path"]}/score-{options["output_prefix"]}{ "-tv" if options["use_train_vocabulary"] else "" }.txt'
labeler = Encoder(options['labels'])
print(labeler)

if options['use_imdb_vocab']:
    include = options['include']
    vocabulary = Encoder.open('movie-review-HW2/aclImdb/imdb.vocab', include=include)
    print(vocabulary)
else:
    vocabulary = None

if options['parse_corpus']:
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
        next(reader, None)
        negations = { negation: token for negation, token in reader }

    with open('src/data/stopwords.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        stopwords = { word for [word] in reader }

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

    train_corpus = Corpus.open(options['train_input'],
        bigrams=options['bigram'],
        vocabulary=vocabulary,
        replacements=replacements,
        expansions=expansions,
        ignored=ignored,
        verbose=True
    )
    print(train_corpus)
    train_corpus.write(options['train_corpus_output'])

    test_corpus = Corpus.open(options['test_input'],
        bigrams=options['bigram'],
        vocabulary=vocabulary,
        replacements=replacements,
        expansions=expansions,
        ignored=ignored,
        verbose=True
    )
    print(test_corpus)
    test_corpus.write(options['test_corpus_output'], verbose=True)

if not options['parse_corpus']:
    train_corpus = Corpus.open(options['train_corpus_output'], frequencies=True, verbose=True)
    print(train_corpus)
    test_corpus = Corpus.open(options['test_corpus_output'], frequencies=True, verbose=True) 
    print(test_corpus)

if not options['use_imdb_vocab'] or options['use_train_vocabulary']:
    vocabulary = Encoder(list(train_corpus.frequencies.keys()))
    print(vocabulary)

model = Model(vocabulary, labeler, log=True)
model.fit(train_corpus, train_corpus.labels(), verbose=True)
print(model)

predictions = model.predict(test_corpus.documents(limit=options['limit_predictions']), verbose=True, debug=False)
score = Metrics.score(test_corpus.labels(limit=options['limit_predictions']), labeler.decode(predictions), labeler)
print(score)
with open(options["results_path"], 'w') as file:
    file.write(f'{score}\n')
    file.write(f'{labeler}\n')
    file.write(f'{vocabulary}\n')
    file.write(f'{train_corpus}\n')
    file.write(f'{test_corpus}\n')
    file.write(f'{model}\n')
    print(score['confusion'], file=file)
    file.write(f'{options}\n')
