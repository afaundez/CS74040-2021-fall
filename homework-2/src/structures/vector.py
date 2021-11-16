class Vector(list):
    def __init__(self, dimensions=None, data=None, default=None, metric='value', name=''):
        self.dimensions = dimensions
        self.name = name
        self.metric = metric
        if data:
            super().__init__(data)
        else:
            super().__init__([default] * len(dimensions))
    
    def max_label_length(self):
        return max(len(str(label)) for label in [self.name, *self.dimensions])
    
    def max_value_cell_length(self):
        return max(len(str(value)) for value in [self.metric, *self])
    
    def __str__(self, default='', filler='-', joint='-+-', pad=' ', separator=' | '):
        header = separator.join([
            str(self.name).ljust(self.max_label_length(), pad),
            str(self.metric).rjust(self.max_value_cell_length(), pad)
        ])
        hline = joint.join([
            str(default).ljust(self.max_label_length(), filler),
            str(default).rjust(self.max_value_cell_length(), filler)
        ])
        rows = [
            separator.join([
                str(label).ljust(self.max_label_length(), pad),
                str(value).rjust(self.max_value_cell_length(), pad)
            ])
            for label, value in zip(self.dimensions, self)
        ]
        return '\n'.join([header, hline, *rows])
