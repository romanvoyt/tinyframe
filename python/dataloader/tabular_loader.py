import pandas as pd

class TabLoader():
    def __init__(self, path, file_type, sep=',', header='infer', index_col=None, columns=None, dtype=None, engine='python', encoding=None, verbose=False):
        self.path = path
        self.file_type = file_type
        self.sep = sep
        self.header = header
        self.index_col = index_col
        self.columns = columns
        self.dtype = dtype
        self.engine = engine
        self.encoding = encoding
        self.verbose = verbose

    def load(self):
        if self.file_type == 'csv':
            df = pd.read_csv(self.path, sep=self.sep, header=self.header, index_col=self.index_col, names=self.columns, dtype=self.dtype, engine=self.engine, encoding=self.encoding)
        else:
            df = pd.read_excel(self.path, header=self.header, index_col=self.index_col, names=self.columns, dtype=self.dtype, engine=self.engine, encoding=self.encoding)
        if self.verbose:
            print(f'Tabular data loaded from {self.path}')
        return df