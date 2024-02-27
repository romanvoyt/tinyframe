class BaseFrame:
    def __init__(self, data=None, columns=None):
        self.data = data
        self.columns = columns
    
    def __repr__(self):
        return str(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, int):
            return self.data[key]
        elif isinstance(key, list):
            return BaseFrame({col: self.data[col] for col in key})
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def head(self, n=5):
        return BaseFrame({col: self.data[col][:n] for col in self.columns})
    
    def tail(self, n=5):
        return BaseFrame({col: self.data[col][-n:] for col in self.columns})
    
    def shape(self):
        return len(self.data), len(self.columns)
    
    def info(self):
        return f"Columns: {self.columns}\nData Types: {self.data.dtypes}"
