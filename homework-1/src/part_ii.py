import math
from lm.corpus import filenameToCorpora, processGrams
from lm.unigram import UnigramModel
from lm.bigram import BigramModel
from lm.smooth_bigram import SmoothBigramModel
from lm.katz_bigram import KatzBigramModel

def preprocess():
    train_corpora = filenameToCorpora('../train.txt', startToken='<s>', stopToken='</s>')
    train_unigrams, train_bigrams = processGrams(train_corpora)
    test_corpora = filenameToCorpora('../test.txt', startToken='<s>', stopToken='</s>')
    test_unigrams, test_bigrams = processGrams(test_corpora)

    test_only_unigrams = { unigram: count for unigram, count in test_unigrams.items() if unigram not in train_unigrams }

    test_only_bigrams = {}
    for condition, word_count in test_bigrams.items():
        if condition not in train_bigrams:
            test_only_bigrams[condition] = word_count
        else:
            for word, count in word_count.items():
                if word not in train_bigrams[condition]:
                    if condition not in test_only_bigrams:
                        test_only_bigrams[condition] = {}
                    test_only_bigrams[condition][word] = count
    
    train_once_unigrams = { word: count for word, count in train_unigrams.items() if count == 1 }

    train_corpora_with_replacing = filenameToCorpora('../train.txt', startToken='<s>', stopToken='</s>', unknownToken='<unk>', unknownWords=set(train_once_unigrams.keys()))
    train_unigrams_with_replacing, train_bigrams_with_replacing = processGrams(train_corpora_with_replacing)

    train_once_test_only_unigrams = set(train_once_unigrams.keys()).union(set(test_only_unigrams.keys()))

    test_corpora_with_replacing = filenameToCorpora('../test.txt', startToken='<s>', stopToken='</s>', unknownToken='<unk>', unknownWords=train_once_test_only_unigrams)
    test_unigrams_with_replacing, test_bigrams_with_replacing = processGrams(test_corpora_with_replacing)
    return train_unigrams_with_replacing, test_only_unigrams, test_unigrams, test_only_bigrams, test_bigrams, train_bigrams_with_replacing

def training(train_unigrams_with_replacing, train_bigrams_with_replacing):
    unigramModel = UnigramModel(train_unigrams_with_replacing, ignoredWords={'<s>'})
    bigramModel = BigramModel(train_unigrams_with_replacing, train_bigrams_with_replacing, ignoreWords={})
    smoothBigramModel = SmoothBigramModel(train_unigrams_with_replacing, train_bigrams_with_replacing, ignoreWords={})
    katzBigramModel = KatzBigramModel(train_unigrams_with_replacing, train_bigrams_with_replacing, ignoreWords={})
    return unigramModel, bigramModel, smoothBigramModel, katzBigramModel


if __name__ == '__main__':
    train_unigrams_with_replacing, test_only_unigrams, test_unigrams, test_only_bigrams, test_bigrams, train_bigrams_with_replacing = preprocess()
    unigramModel, bigramModel, smoothBigramModel, katzBigramModel = training(train_unigrams_with_replacing, train_bigrams_with_replacing)


    excluded_unigrams =  {'<s>'}
    train_unigrams_with_replacing_without_start = set(train_unigrams_with_replacing.keys()) - excluded_unigrams
    print(f'There are {len(train_unigrams_with_replacing_without_start)} word types (unique words) in the training corpus.')

    excluded_unigrams =  {'<s>'}

    train_unigrams_with_replacing_without_start = sum([ value for word, value in train_unigrams_with_replacing.items() if word not in excluded_unigrams ])
    print(f'There are {train_unigrams_with_replacing_without_start} word tokens in the training corpus.')

    excluded_unigrams =  {'<s>'}
    test_only_unigrams_without_start = { word: count for word, count in test_only_unigrams.items() if word not in excluded_unigrams }
    test_unigrams_without_start = { word: count for word, count in test_unigrams.items() if word not in excluded_unigrams }

    excluded_unigrams =  {'<s>'}

    test_only_bigrams_types_without_start_count = sum(len(word_count.keys()) for condition, word_count in test_only_bigrams.items() if condition not in excluded_unigrams)
    print('\\item Bigrams types only in test corpus:', test_only_bigrams_types_without_start_count)
    test_bigrams_types_without_start_count = sum(len(word_count.keys()) for condition, word_count in test_bigrams.items() if condition not in excluded_unigrams)
    print('\\item Bigrams types in test corpus:', test_bigrams_types_without_start_count)
    print('\\item \\textbf{Percentage ofbigrams types only in test corpus}:', test_only_bigrams_types_without_start_count / test_bigrams_types_without_start_count * 100)

    test_only_bigrams_words_without_start_count = sum(sum(word_count.values()) for condition, word_count in test_only_bigrams.items() if condition not in excluded_unigrams)
    print('\\item Bigrams tokes only in test corpus:', test_only_bigrams_words_without_start_count)
    test_bigrams_words_without_start_count = sum(sum(word_count.values()) for condition, word_count in test_bigrams.items() if condition not in excluded_unigrams)
    print('\\item Bigrams tokens in test corpus:', test_bigrams_words_without_start_count)
    print('\\item \\textbf{Percentage bigrams tokens only in test}:', test_only_bigrams_words_without_start_count / test_bigrams_words_without_start_count * 100)

    # unigramModel = UnigramModel(train_unigrams_with_replacing, ignoredWords={'<s>'})
    unigramLogOutput = unigramModel.sentenceMLE('I look forward to hearing your reply .', verbose=True, log=True)
    unigramOutput = unigramModel.sentenceMLE('I look forward to hearing your reply .', verbose=True)

    # bigramModel = BigramModel(train_unigrams_with_replacing, train_bigrams_with_replacing, ignoreWords={})
    bigramLogOutput = bigramModel.sentenceMLE('I look forward to hearing your reply .', verbose=True, log=True)

    # smoothBigramModel = SmoothBigramModel(train_unigrams_with_replacing, train_bigrams_with_replacing, ignoreWords={})
    smoothBigramLogOutput = smoothBigramModel.sentenceMLE('I look forward to hearing your reply .', verbose=True, log=True)
    output = smoothBigramModel.sentenceMLE('I look forward to hearing your reply .', verbose=True)

    # katzBigramModel = KatzBigramModel(train_unigrams_with_replacing, train_bigrams_with_replacing, ignoreWords={})
    katzBigramLogOutput = katzBigramModel.sentenceMLE('I look forward to hearing your reply .', verbose=True, log=True)
    katzBigramOutput = katzBigramModel.sentenceMLE('I look forward to hearing your reply .', verbose=True)

    for output, title in [
        (unigramLogOutput, 'Unigram Model'),
        (bigramLogOutput, 'Bigram Model'),
        (smoothBigramLogOutput, 'Smooth Bigram Model'),
        (katzBigramLogOutput, 'Katz Bigram Model')]:

        *_, M, log_p = output
        l = 1/M * log_p
        pp = math.pow(2, -l)
        print('\\paragraph{%s}' % title)
        print('\\begin{equation}')
        print('\\begin{split}')
        print('M &= ' + str(M) + ' \\\\')
        print('\\log_{2} (P(S)) &= ' + str(log_p) + ' \\\\')
        print('l &= \\frac{1}{M} * \log_{2} = \\frac{1}' + str(M) + ' \\times ' + str(log_p) + ' = ' + str(l) +' \\\\')
        print('Perplexity(S) &= 2^{-l} = ' + str(pp) + '\\\\')
        print('\\end{split}')
        print('\\end{equation}')
