import math
from lm.bigram import BigramModel
class KatzBigramModel(BigramModel):
    def __init__(self, unigrams, bigrams, ignoreWords={}):
        BigramModel.__init__(self, unigrams, bigrams, ignoreWords=ignoreWords)

        self.bigrams_star = {}
        for condition, word_count in self.bigrams.items():
            if condition not in self.bigrams_star:
                self.bigrams_star[condition] = {}
            for word, count in word_count.items():
                if word not in self.bigrams_star[condition]:
                    self.bigrams_star[condition][word] = 0
                self.bigrams_star[condition][word] = count - 0.5
        self.A = {}
        self.B = {}
        self.P_B = {}
        self.P_ML = { word: self.unigramMLE(word)[-1] for word in self.unigrams }
    
    def trainB(self, condition):
        self.A[condition] = {}
        self.B[condition] = {}
        for unigram in self.unigrams:
            if condition in self.bigrams and unigram in self.bigrams[condition]:
                self.A[condition][unigram] = self.bigrams[condition][unigram]
        for unigram, count in self.unigrams.items():
            if unigram in self.A[condition]:
                continue
            self.B[condition][unigram] = count
        self.P_B[condition] = sum(self.P_ML[word] for word in self.B[condition])

    def trainBs(self, bigrams):
        for condition, unigrams in bigrams.items():
            for word in unigrams:
                if condition in self.bigrams and word in self.bigrams[condition]:
                    continue
                self.trainB(condition)

    def bigramMLE(self, word, condition, log=False, verbose=False):
        alpha_steps = None
        steps = ['P(\\texttt{' + condition + '} \mid \\texttt{' + word + '})']
        if condition in self.bigrams_star and word in self.bigrams_star[condition]:
            bigramCount = self.bigrams_star[condition][word]
            unigramCount = self.unigrams[condition]
            conditionalProbability = bigramCount / unigramCount
            steps.append('\\frac{count^{*}(\\texttt{' + condition + '} , \\texttt{' + word + '})}{count(\\texttt{' + condition + '})}')
            steps.append('\\frac{' + str(bigramCount) +  '}{' + str(unigramCount) + '}')
        else:
            if condition not in self.P_B:
                self.trainB(condition)
            
            alpha_numerator = sum(self.bigrams_star[condition].values())
            alpha_denominator = self.unigrams[condition]
            alpha_condition = 1 - alpha_numerator / alpha_denominator
            alpha_steps = [
                '\\alpha_{\\texttt{' + condition + '}}',
                '1 - \\frac{\\Sigma_{w} count^{*}(\\texttt{' + condition + '} , \\texttt{' + word + '})}{count(\\texttt{' + condition + '})}',
                '1 - \\frac{' + str(alpha_numerator) +  '}{' + str(alpha_denominator) + '}',
                alpha_condition
            ]
            *_, wordProbability = self.unigramMLE(word, log=False)
            wordProbabilitiesSum = self.P_B[condition]
            conditionalProbability = alpha_condition * wordProbability / wordProbabilitiesSum
            steps.append('\\alpha_{\\texttt{' + condition + '}} \\times \\frac{P_{ML}(\\texttt{' + word + '})}{\\Sigma_{w \\in B_{\\texttt{' + condition + '}}} P(\\texttt{w})}')
            steps.append(str(alpha_condition) + '\\times \\frac{' + str(wordProbability) +  '}{' + str(wordProbabilitiesSum) + '}')
        
        if log:
            conditionalProbability = math.log(conditionalProbability, 2) if conditionalProbability > 0 else -math.inf
        if log:
            steps = [ '\log_{2} (' + step + ')' for step in steps ]
        steps.append(conditionalProbability)
        if verbose:
            if alpha_steps:
                print(alpha_steps[0] + ' =&\\ ', ' = '.join([ str(step) for step in alpha_steps[1:] ]), ' \\\\')
            print(steps[0] + ' =&\\ ', ' = '.join([ str(step) for step in steps[1:] ]), ' \\\\')
        return steps
