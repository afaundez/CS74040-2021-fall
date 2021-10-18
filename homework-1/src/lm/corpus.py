from itertools import tee

def loadSentences(filename):
    with open(filename, 'r') as f:
        corpora = f.read().strip()
        return [ sentence.strip().split(' ') for sentence in corpora.split('\n') ]

def padSentence(sentence, startToken='<s>', stopToken='</s>'):
    return [startToken] + sentence + [stopToken]

def lowerSentence(sentence):
    return [ word.lower() for word in sentence ]

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def sentenceToWords(sentence, startToken='<s>', stopToken='</s>', unknownToken='<unk>', unknownWords=set(), knownWords=set(), ignoreWords=set()):
    words = sentence.strip().lower().split(' ')
    isUnknown = lambda word: word in unknownWords or (len(knownWords) > 0 and word not in knownWords)
    words = [ unknownToken if isUnknown(word) else word for word in words ]
    words =  [startToken, *words, stopToken]
    words = [ word for word in words if word not in ignoreWords ]
    return words

def replaceWordsInSentence(sentence, wordsForReplacing=[], replacingToken='<unk>'):
    return [
        word if word not in wordsForReplacing else replacingToken
        for word in sentence
    ]

def filenameToCorpora(filename, startToken='<s>', stopToken='</s>', unknownToken='<unk>', unknownWords=[]):
    return [
        replaceWordsInSentence(
            lowerSentence(
                padSentence(sentence, startToken, stopToken)),
            wordsForReplacing=unknownWords, replacingToken=unknownToken)
        for sentence in loadSentences(filename)
    ]

def processGrams(corpora):
    unigrams = {}
    bigrams = {}
    for sentence in corpora:
        for (condition, word) in pairwise(sentence):
            if condition not in bigrams:
                bigrams[condition] = {}
            if word not in bigrams[condition]:
                bigrams[condition][word] = 0
            bigrams[condition][word] += 1
        for token_type in sentence:
            if token_type not in unigrams:
                unigrams[token_type] = 0
            unigrams[token_type] += 1
    return unigrams, bigrams