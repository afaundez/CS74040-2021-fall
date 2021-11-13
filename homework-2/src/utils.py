class Utils:
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