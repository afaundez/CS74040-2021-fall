# Movie review classification using Naïve Bayes

## Prerequisites

* [python 3](https://www.python.org/downloads/). Tested on 3.8.9 (macOS/system) and 3.10.0 (macOS/wpyenv).

## Pre-process

```sh
python3 pre-process.py \
    --training-file='movie-review-HW2/aclImdb/train/**/*.txt' \
    --test-file='movie-review-HW2/aclImdb/test/**/*.txt' \
    --output-path=movie-review-HW2/aclImdb \
    --vocabulary-file=movie-review-HW2/aclImdb/imdb.vocab \
    --add-label=pos --add-label=neg \
    --ngrams=1
```

## Train and Predict

```sh
python3 NB.py \
    --training-file=movie-review-HW2/aclImdb/train-1grams.NB \
    --test-file=movie-review-HW2/aclImdb/test-1grams.NB \
    --output-file=movie-review-HW2/aclImdb/test-1grams.NB.output \
    --vocabulary-file=movie-review-HW2/aclImdb/imdb.vocab \
    --use-train-vocabulary
```