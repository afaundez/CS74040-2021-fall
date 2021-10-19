# HW 1 NLP

Be sure of run everything with python 3.

## HW answers
All the answers for the homework are at `part_i.py` and `part_ii.py`. They output mostly LaTeX code.

```sh
cd homework-1/src
python3 part_i.py
python3 part_ii.py
```

There is also a notebook with everything running.

## Models

To use a model, first you need to load a corpus and process the grams:

```python
from model import filenameToCorpora, processGrams
corpora = filenameToCorpora('../data/train.txt', startToken='<s>', stopToken='</s>', unknownToken='<unk>', unknownWords=[])
unigrams, bigrams = processGrams(corpora)
```

Then you can use it with the models the models:
```python
from unigram_model import UnigramModel
model = UnigramModel(unigrams, ignoredWords={'<s>'})
*output, value = model.sentenceMLE('I look forward to hearing your reply .')
print(value)
*output, value = model.unigramMLE('look')
print(value)

from bigram_model import BigramModel
model = BigramModel(unigrams, bigrams)
*output, value = model.sentenceMLE('I look forward to hearing your reply .')
print(value)
*output, value = model.bigramMLE('reply', 'your')
print(value)

from smooth_bigram_model import SmoothBigramModel
model = SmoothBigramModel(unigrams, bigrams)
*output, value = model.sentenceMLE('I look forward to hearing your reply .')
print(value)
*output, value = model.bigramMLE('look', 'I')
print(value)

from katz_bigram_model import KatzBigramModel
model = KatzBigramModel(unigrams, bigrams)
*output, value = model.sentenceMLE('I look forward to hearing your reply .')
print(value)
*output, value = model.bigramMLE('look', 'I')
print(value)
```

Just be sure that run the code at the `src` directory. A small working example is in `part_i.py`. A more complete example is in `part_ii.py`, but the output is not very useful, since it's mostly LaTeX code.

## Model options

When calculating a probability (using sentenceMLE or bigramMLE), you can specify the options:

- `log=True` to return the log of the probability
- `verbose=True` to print the LaTeX output
