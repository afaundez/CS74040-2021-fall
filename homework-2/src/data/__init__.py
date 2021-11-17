import csv
import pathlib

def text_helpers(labeler):

    filename = pathlib.Path(__file__).parent / 'acronyms.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        acronyms = { acronym: meaning for acronym, meaning in reader }

    filename = pathlib.Path(__file__).parent / 'smileys.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        smileys = { smiley: f'{labeler.decode(int(bias))}' for smiley, bias in reader }

    filename = pathlib.Path(__file__).parent / 'positive-words.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        positive_words = { positive_word: f'{labeler.decode(int(bias))}' for positive_word, bias in reader }

    filename = pathlib.Path(__file__).parent / 'negative-words.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        negative_words = { negative_word: f'{labeler.decode(int(bias))}' for negative_word, bias in reader }

    filename = pathlib.Path(__file__).parent / 'negation.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        negations = { negation: token for negation, token in reader }

    filename = pathlib.Path(__file__).parent / 'stopwords.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        stopwords = { word for [word] in reader }
    
    return acronyms, smileys, positive_words, negative_words, negations, stopwords