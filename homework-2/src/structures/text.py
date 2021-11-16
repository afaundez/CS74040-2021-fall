import html
import re
from src.utils.decorators import debug
from src.utils.processors import HTMLProcessor

URL_PATTERN = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
EMAIL_PATTERN = re.compile(r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')
TOKEN_PATTERN = re.compile(r'\S+')

class Text(str):
    @debug
    def deurlize(self, **_):
        self = Text(URL_PATTERN.sub(r'||url||', self))
        return self
    
    @debug
    def deemailize(self, **_):
        self = Text(EMAIL_PATTERN.sub(r'||email||', self))
        return self
    
    @debug
    def descape(self, **_):
        self = Text(html.unescape(self))
        return self
    
    @debug
    def dehtmlize(self, **_):
        processor = HTMLProcessor()
        processor.feed(self)
        self = Text(processor.text)
        return self
    
    @debug
    def deunicodify(self, **_):
        self = Text(self.encode('ascii','ignore').decode('utf-8'))
        return self