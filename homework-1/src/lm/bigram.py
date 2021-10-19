import math
from .unigram import UnigramModel
from .corpus import sentenceToWords, pairwise
class BigramModel(UnigramModel):
    def __init__(self, unigrams, bigrams, ignoreWords={}):
        super().__init__(unigrams, ignoredWords=ignoreWords)
        self.bigrams = bigrams
    
    def bigramMLE(self, word, condition, log=False, verbose=False):
        if condition in self.bigrams and word in self.bigrams[condition]:
            bigramCount = self.bigrams[condition][word]
            unigramCount = self.unigrams[condition]
            conditionalProbability = bigramCount / unigramCount
        else:
            bigramCount = 0
            unigramCount = 0
            conditionalProbability = 0
        if log:
            conditionalProbability = math.log(conditionalProbability, 2) if conditionalProbability > 0 else -math.inf
        steps = [
            'P(\\texttt{' + condition + '} \mid \\texttt{' + word + '})',
            '\\frac{count(\\texttt{' + condition + '} , \\texttt{' + word + '})}{count(' + condition + ')}',
            '\\frac{' + str(bigramCount) +  '}{' + str(unigramCount) + '}'
        ]
        if log:
            steps = [ '\log_{2} (' + step + ')' for step in steps ]
        steps.append(conditionalProbability)
        if verbose:
            print(steps[0] + ' =&\\ ', ' = '.join([ str(step) for step in steps[1:] ]), ' \\\\')
        return steps

    def sentenceMLE(self, sentence, log=False, verbose=False):
        if verbose:
            print("\\begin{equation}\\begin{split}")
            print('S =&\\ \\texttt{' + sentence + '} \\\\')
        
        words = sentenceToWords(sentence, knownWords=set(self.unigrams.keys()), ignoreWords=self.ignoredWords)

        wordProbabilities = [ self.bigramMLE(word, condition, log=log, verbose=verbose) for condition, word in pairwise(words) ]
        sentenceProbability = self.operatorSumOrProd([ wordProbability for *_, wordProbability in wordProbabilities ], log=log)

        steps = [
            'P(S)',
            'P(\\texttt{'+ sentence + '})',
            'P(' + ', '.join([ '\\texttt{' + word + '}' for word in words]) +')'
        ]
        if log:
            steps = [ '\log_{2} (' + step + ')' for step in steps ]
        steps.append(self.operatorPlusOrTimes(log).join([ wordProbabilityKey for wordProbabilityKey, *_ in wordProbabilities ]))
        steps.append(self.operatorPlusOrTimes(log).join([ str(wordProbability) for *_, wordProbability in wordProbabilities ]))
        steps.append(len(wordProbabilities))
        steps.append(sentenceProbability)

        if verbose:
            print(steps[0], '&=', ' =&\\ '.join([f'{step} \\\\' for step in steps[1:]]))
            print("\\end{split}\\end{equation}")
        return steps
