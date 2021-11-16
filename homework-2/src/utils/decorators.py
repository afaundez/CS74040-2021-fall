def incremental(label):
    def wrapper(decorable):
        def inner(*args, **kwargs):
            for index, item in enumerate(decorable(*args, **kwargs)):
                if 'verbose' in kwargs and kwargs['verbose'] and index != 0 and index % 1000 == 0:
                    print(f'{index} {label}')
                yield item
        return inner
    return wrapper

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