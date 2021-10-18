import math
from .corpus import sentenceToWords

class UnigramModel:
    def __init__(self, unigrams, ignoredWords={}):
        self.ignoredWords = ignoredWords
        self.unigrams = { unigram: count for unigram, count in unigrams.items() if unigram not in self.ignoredWords }
    
    def operatorSumOrProd(self, values, log=False):
        return sum(values) if log else math.prod(values)
    
    def operatorPlusOrTimes(self, log=False):
        return ' + ' if log else ' \\times '
    
    def unigramMLE(self, word, log=False, verbose=False):
        denominator = sum(self.unigrams.values())
        if word in self.unigrams:
            numerator = self.unigrams[word]
            wordProbability = numerator / denominator
        else:
            numerator = 0
            wordProbability = 0
        if log:
            wordProbability = math.log(wordProbability, 2) if wordProbability > 0 else -math.inf
        
        steps = [
            'P(\\texttt{'+ word + '})',
            '\\frac{count(\\texttt{'+ word + '})}{count()}',
            '\\frac{' + str(numerator) +  '}{' + str(denominator) + '}'
        ]
        if log:
            steps = [ '\log_{2} (' + step + ')' for step in steps ]
        steps.append(wordProbability)

        if verbose:
            print(steps[0] + ' &= ', ' = '.join([ str(step) for step in steps[1:] ]), ' \\\\')
        return steps

    def sentenceMLE(self, sentence, log=False, verbose=False):
        if verbose:
            print("\\begin{equation}\\begin{split}")
            print('S &= \\texttt{' + sentence + '} \\\\')

        words = sentenceToWords(sentence, knownWords=set(self.unigrams.keys()), ignoreWords=self.ignoredWords)
        wordProbabilities = [ self.unigramMLE(word, log=log, verbose=verbose) for word in words if word not in self.ignoredWords ]
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
            print(steps[0], '&=', ' &= '.join([f'{step} \\\\' for step in steps[1:]]))
            print("\\end{split}\\end{equation}")
        return steps