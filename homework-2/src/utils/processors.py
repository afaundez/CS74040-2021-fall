from html.parser import HTMLParser

class HTMLProcessor(HTMLParser):
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


def argmax(iterable):
    max_index = None
    max_value = None
    for index, value in enumerate(iterable):
        if max_value is None or value > max_value:
            max_index = index
            max_value = value
    return max_index

def loggify(text, log=True):
    if log:
        return f'log_2({text})'
    return text