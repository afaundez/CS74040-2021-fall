class Encoder(set):
    def open(path, include=[], exclude=[]):
        with open(path, mode='r', encoding='utf-8') as file:
            tokens = [ line.strip() for line in file ]
            return Encoder(tokens, include, exclude)
   
    def __init__(self, values=[], include=[], exclude=[]):
        super().__init__([ value for value in values + include if value not in exclude ])
        self.indexes = { value: index for index, value in enumerate(self) }
        self.values = { index: value for value, index in self.indexes.items() }
        self.size = len(self)
    
    def labels(self):
        return list(self)

    def encode(self, value):
        return self.indexes[value]
    
    def decode(self, index_or_indexes):
        if isinstance(index_or_indexes, int):
            return self.values[index_or_indexes] 
        return [ self.values[index] for index in index_or_indexes ]
    
    def __str__(self) -> str:
        if len(self) < 10:
            sample = list(self)
        else:
            sample = []
            for index, token in enumerate(self):
                sample.append(token)
                if index == 5:
                    break
        return f'Encoder(tokens={len(self)}, sample={sample})'
