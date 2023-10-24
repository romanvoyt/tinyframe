import pandas as pd

class TabLoader():
    def __init__(self, path, file_type, spreadsheet_name='', sep=',', index_col=None, columns=None, dtype=None, engine='python', encoding=None, verbose=False):
        self.path = path
        self.file_type = file_type
        self.spreadsheet_name = spreadsheet_name
        self.sep = sep
        self.index_col = index_col
        self.columns = columns
        self.dtype = dtype
        self.engine = engine
        self.encoding = encoding
        self.verbose = verbose

    def load_spreadsheet(self):
        assert self.file_type in ['csv', 'xlsx'], 'File type must be either csv or xlsx'
        
        if self.file_type == 'csv':
            df = pd.read_csv(self.path, sep=self.sep, index_col=self.index_col, names=self.columns, dtype=self.dtype, engine=self.engine)
        else:
            df = pd.read_excel(self.path,sheet_name=self.spreadsheet_name, index_col=self.index_col, names=self.columns, dtype=self.dtype)
        if self.verbose:
            print(f'Tabular data loaded from {self.path}')
        return df