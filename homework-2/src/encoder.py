class Encoder(set):
    def open(path):
        with open(path, mode='r', encoding='utf-8') as file:
            tokens = [ line.strip() for line in file ]
            return Encoder(tokens)
    
    def __init__(self, values=[]):
        super().__init__(values)
        self.indexes = { value: index for index, value in enumerate(self) }
        self.values = { index: value for value, index in self.indexes.items() }
        self.size = len(self)

    def encode(self, value):
        return self.indexes[value]
    
    def decode(self, index_or_indexes):
        if isinstance(index_or_indexes, int):
            return self.values[index_or_indexes] 
        return [ self.values[index] for index in index_or_indexes ]
    
    def summary(self, size=10):
        print({ 'Encoder' : { 'size': self.size, 'indexes': self.indexes } })
