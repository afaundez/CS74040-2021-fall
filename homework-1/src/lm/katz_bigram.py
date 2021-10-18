import math
from lm.bigram import BigramModel

class KatzBigramModel(BigramModel):
    def __init__(self, unigrams, bigrams, ignoreWords={}):
        BigramModel.__init__(self, unigrams, bigrams, ignoreWords=ignoreWords)

        self.bigrams_star = {}
        # self.leftovers = {}
        for condition, word_count in self.bigrams.items():
            # self.leftovers[condition] = 1
            if condition not in self.bigrams_star:
                self.bigrams_star[condition] = {}
            for word, count in word_count.items():
                if word not in self.bigrams_star[condition]:
                    self.bigrams_star[condition][word] = 0
                self.bigrams_star[condition][word] = count - 0.5
                # self.leftovers[condition] += 0.5

    def bigramMLE(self, word, condition, log=False, verbose=False):
        if condition in self.bigrams_star:
            A_condition = set(self.bigrams_star[condition].keys())
        else:
            A_condition = set()

        alpha_steps = None
        steps = ['P(\\texttt{' + condition + '} \mid \\texttt{' + word + '})']
        B_condition = set(self.unigrams.keys()) - A_condition
        if word in A_condition:
            bigramCount = self.bigrams_star[condition][word]
            unigramCount = self.unigrams[condition]
            conditionalProbability = bigramCount / unigramCount
            steps.append('\\frac{count^{*}(\\texttt{' + condition + '} , \\texttt{' + word + '})}{count(\\texttt{' + condition + '})}')
            steps.append('\\frac{' + str(bigramCount) +  '}{' + str(unigramCount) + '}')
        else:
            alpha_numerator = sum(self.bigrams_star[condition].values())
            alpha_denominator = self.unigrams[condition]
            alpha_condition = 1 - alpha_numerator / alpha_denominator
            alpha_steps = [
                '\\alpha_{\\texttt{' + condition + '}}',
                '1 - \\frac{\\Sigma_{w} count^{*}(\\texttt{' + condition + '} , \\texttt{' + word + '})}{count(\\texttt{' + condition + '})}',
                '1 - \\frac{' + str(alpha_numerator) +  '}{' + str(alpha_denominator) + '}',
                alpha_condition
            ]
            *_, wordProbability = self.unigramMLE(word, log=log)
            wordProbabilitiesSum = sum(self.unigramMLE(unigram, log=log)[-1] for unigram in self.unigrams if unigram in B_condition)
            conditionalProbability = alpha_condition * wordProbability / wordProbabilitiesSum
            steps.append('\\alpha_{\\texttt{' + condition + '}} \\times \\frac{P(\\texttt{' + word + '})}{\\Sigma_{w \\in B} P(\\texttt{' + word + '})}')
            steps.append(str(alpha_condition) + '\\times \\frac{' + str(wordProbability) +  '}{' + str(wordProbabilitiesSum) + '}')
        
        if log:
            conditionalProbability = math.log(conditionalProbability) if conditionalProbability > 0 else -math.inf
        if log:
            steps = [ '\log_{2} (' + step + ')' for step in steps ]
        steps.append(conditionalProbability)
        if verbose:
            if alpha_steps:
                print(alpha_steps[0] + ' &= ', ' = '.join([ str(step) for step in alpha_steps[1:] ]), ' \\\\')
            print(steps[0] + ' &= ', ' = '.join([ str(step) for step in steps[1:] ]), ' \\\\')
        return steps