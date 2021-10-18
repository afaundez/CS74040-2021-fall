import math
from .bigram import BigramModel

class SmoothBigramModel(BigramModel):
    def bigramMLE(self, word, condition, log=False, verbose=False):
        if condition in self.bigrams and word in self.bigrams[condition]:
            bigramCount = self.bigrams[condition][word]
        else:
            bigramCount = 0
        if condition in self.unigrams:
            unigramCount = self.unigrams[condition]
        else:
            unigramCount = 0
        vocab_size = len(self.unigrams.keys())
        conditionalProbability = (bigramCount + 1) / (unigramCount + vocab_size)
        if log:
            conditionalProbability = math.log(conditionalProbability) if conditionalProbability > 0 else -math.inf
        steps = [
            'P(\\texttt{' + condition + '} \mid \\texttt{' + word + '})',
            '\\frac{count^{*}(\\texttt{' + condition + '} , \\texttt{' + word + '}) + 1}{count(' + condition + ') + |V|}',
            '\\frac{' + str(bigramCount) +  ' + 1}{' + str(unigramCount) + ' + ' + str(vocab_size) + '}'
        ]
        if log:
            steps = [ '\log_{2} (' + step + ')' for step in steps ]
        steps.append(conditionalProbability)
        if verbose:
            print(steps[0] + ' &= ', ' = '.join([ str(step) for step in steps[1:] ]), ' \\\\')
        return steps
