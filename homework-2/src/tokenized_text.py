import re
import html
from html.parser import HTMLParser
from src.utils.generators import split, expand, replace, ignore, check

URL_PATTERN = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
EMAIL_PATTERN = re.compile(r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')
TOKEN_PATTERN = re.compile(r'\S+')

class HTMLFilter(HTMLParser):
    text = ''
    def handle_data(self, data):
        self.text += data
    
    def handle_starttag(self, tag, attrs):
        self.text += ' '
        return super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.text += ' '
        return super().handle_endtag(tag)
    
    def handle_startendtag(self, tag    , attrs):
        self.text += ' '
        return super().handle_startendtag(tag, attrs)

def debug(target):
    def wrapper(*args, **kwargs):
        if 'debug' in kwargs and kwargs['debug']:
            print('PRE', target.__name__, '\n', *args)
        result = target(*args, **kwargs)
        if 'debug' in kwargs and kwargs['debug']:
            print('POST ', target.__name__, '\n', result)
        if 'debug' in kwargs and kwargs['debug']:
            print('END ', target.__name__)
        return result

    return wrapper

class Text(str):
    @debug
    def deurlize(self, debug=False):
        self = Text(URL_PATTERN.sub(r'||url||', self))
        return self
    
    @debug
    def deemailize(self, debug=False):
        self = Text(EMAIL_PATTERN.sub(r'||email||', self))
        return self
    
    @debug
    def descape(self, debug=False):
        self = Text(html.unescape(self))
        return self
    
    @debug
    def dehtmlize(self, debug=False):
        filter = HTMLFilter()
        filter.feed(self)
        self = Text(filter.text)
        return self
    
    @debug
    def deunicodify(self, debug=False):
        self = Text(self.encode('ascii','ignore').decode('utf-8'))
        return self

class Tokenizer(map):
    def __new__(cls, text, vocabulary=None, expansions=[], replacements=[], ignored=[], tag_marker='||', **kwargs):
        text = Text(text) \
            .deurlize(debug=debug) \
            .deemailize(debug=debug) \
            .descape(debug=debug) \
            .dehtmlize(debug=debug) \
            .deunicodify(debug=debug) \
            .lower() \
            .strip()

        tokens = split(text, TOKEN_PATTERN)
        tokens = expand(tokens, expansions)
        tokens = replace(tokens, replacements)
        tokens = ignore(tokens, ignored)
        tokens = check(tokens, vocabulary)
        return super().__new__(cls, lambda x: x, tokens)

class BagOfWords(dict):
    def __init__(self, text_or_frequencies, vocabulary=None, expansions={}, replacements={}, ignored=[], **kwargs):
        if isinstance(text_or_frequencies, dict):
            super().__init__(text_or_frequencies)
        else:
            tokens = Tokenizer(text_or_frequencies, expansions=expansions, vocabulary=vocabulary, replacements=replacements, ignored=ignored, **kwargs)
            for token in tokens:
                if token not in self:
                    self[token] = 0
                self[token] += 1
    
    def open(path, *args, **kwargs):
        with open(path, 'r') as file:
            return BagOfWords(file.read(), *args, **kwargs)

    def merge(self, other):
        for token, count in other.items():
            if token not in self:
                self[token] = 0
            self[token] += count
        return self
    
    def write(self, filename):
        with open(filename, 'w') as f:
            for token, count in self.items():
                f.write(f'{token} {count}\n')



    