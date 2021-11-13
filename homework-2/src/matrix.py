class Matrix(list):
    def __init__(self, rows, cols, data=None, default=None, name=''):
        self.rows = rows
        self.cols = cols
        self.name = name
        if data is None:
            data = [ [default] * len(self.cols) for _ in self.rows]
        self.extend(data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return super().__getitem__(row).__getitem__(col)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            super().__getitem__(row).__setitem__(col, value)
        else:
            super().__setitem__(key, value)
    
    def col_max_cell_lenght(self):
        max_cell_lenghts = [ len(col) for col in self.cols ]
        for row in self:
            for col, cell in enumerate(row):
                max_cell_lenghts[col] = max(max_cell_lenghts[col], len(str(cell)))
        return max_cell_lenghts
    
    def row_label_max_lenght(self):
        return max([ len(str(cell)) for cell in [self.name, *self.rows] ])

    
    def __str__(self, default='', filler='-', joint='-+-', pad=' ', separator=' | '):
        cell_lengths = [self.row_label_max_lenght()] + self.col_max_cell_lenght()
        header = separator.join([
            str(cell).rjust(lenght, pad)
            for cell, lenght in zip([self.name, *self.cols], cell_lengths)
        ])
        hline = joint.join([
            default.rjust(length, filler) for length in cell_lengths
        ])
        rows = [
            separator.join([
                str(cell).rjust(length, pad)
                for cell, length in zip([row_label, *self[row]], cell_lengths)
            ])
            for row, row_label in enumerate(self.rows)
        ]
        return '\n'.join([header, hline, *rows])
