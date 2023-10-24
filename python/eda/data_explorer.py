import pandas as pd

class DataExplorer():
    def __init__(self, df):
        self.df = df
    
    def profile(self):
        # profiler = ProfileReport(self.df, title='Pandas Profiling Report', explorative=True)
        # profiler.to_file('profile.html')
        assert type(self.df) == pd.core.frame.DataFrame, 'Dataframe is not a pandas dataframe'
        assert self.df is not None, 'Dataframe is empty'
        
        print(self.df.info(verbose=True))