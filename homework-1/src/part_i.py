from lm.corpus import filenameToCorpora, processGrams
from lm.smooth_bigram import SmoothBigramModel

if __name__ == '__main__':
    p1_corpora = filenameToCorpora('../p1.txt', startToken='<s>', stopToken='</s>')
    p1_unigrams, p1_bigrams = processGrams(p1_corpora)
    p1Model = SmoothBigramModel(p1_unigrams, p1_bigrams, ignoreWords={})
    output = p1Model.bigramMLE('sam', 'am', log=False, verbose=True)