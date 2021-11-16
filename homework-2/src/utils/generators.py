import re
from src.structures.text import Text

TOKEN_PATTERN = re.compile(r'\S+')
PUNCTUATION_REGEX = re.compile(r'([\\\|¦\[\]\{\}$&®@%=~«»…½¡£₤§^`´¨#+.+,:;!?()/<>\-\+\*_"“”‘’\x08\x80-\xff\'])')
HYPEN_REGEX = re.compile(r'(\w+)(-{2,})(\w+)')


def tokenize(text, vocabulary=None, expansions=[], replacements=[], ignored=[], **kwargs):
    text = Text(text) \
        .deurlize(**kwargs) \
        .deemailize(**kwargs) \
        .descape(**kwargs) \
        .dehtmlize(**kwargs) \
        .deunicodify(**kwargs) \
        .lower() \
        .strip()

    tokens = split(text)
    tokens = expand(tokens, expansions)
    tokens = replace(tokens, replacements)
    tokens = ignore(tokens, ignored)
    tokens = check(tokens, vocabulary)
    return tokens

def check(tokens, vocabulary=None, punctuation_pattern=PUNCTUATION_REGEX, hyphen_pattern=HYPEN_REGEX, token_pattern=TOKEN_PATTERN):
    if isinstance(tokens, str):
        tokens = [tokens]
    for token in tokens:
        if token != '':
            if vocabulary is None or token in vocabulary:
                yield token
            else:
                token = re.sub(punctuation_pattern, r' ', token)
                token = re.sub(hyphen_pattern, r'\1 \2 \3', token)
                yield from starsplit(token, token_pattern)

def starsplit(iterable, pattern=TOKEN_PATTERN):
    if isinstance(iterable, str):
        yield from split(iterable, pattern)
    else:
        for item in iterable:
            yield from split(item, pattern)

def split(splittable, pattern=TOKEN_PATTERN):
    tokenize = lambda match: match.group()
    yield from map(tokenize, re.finditer(pattern, splittable))

def expand(iterable, expansions={}, token_pattern=TOKEN_PATTERN):
    expand = lambda token: expansions[token] if token in expansions else token
    yield from starsplit(map(expand, iterable), token_pattern)

def replace(iterable, replacements={}):
    replace = lambda token: replacements[token] if token in replacements else token
    yield from map(replace, iterable)

def ignore(iterable, ignored=[]):
    ignored = set(ignored)
    ignore = lambda token: token not in ignored
    yield from filter(ignore, iterable)

def allow(iterable, allowed=[]):
    allowed = set(allowed)
    allowed = lambda token: token in allowed
    yield from filter(allowed, iterable)