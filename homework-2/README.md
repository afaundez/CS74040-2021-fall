# Movie review classification using Naïve Bayes

## Prerequisites

* [python 3](https://www.python.org/downloads/). Tested on 3.8.9 (macOS/system) and 3.10.0 (macOS/pyenv).

## Pre-process

```sh
python3 pre-process.py \
    --labels pos,neg \
    --input-train='movie-review-HW2/aclImdb/train/**/*.txt' \
    --input-test='movie-review-HW2/aclImdb/test/**/*.txt' \
    --output-path=movie-review-HW2/aclImdb \
    --pre-process=acronym,smileys,positive-words,negative-words,negations,stopwords \
    --ngrams=1 \
    --use-train-vocabulary=True \
    --use-imdb-vocab=True \
    --extra-vocabulary=''
```

## Train and Predict

```sh
python3 NB.py \
    --input-train=movie-review-HW2/aclImdb/train-1grams.NB \
    --input-test=movie-review-HW2/aclImdb/test-1grams.NB \
    --use-train-vocabulary=True \
    --use-imdb-vocab=True \
    --extra-vocabulary=''
```