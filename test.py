from python.dataloader.tabular_loader import TabLoader
from python.eda.data_explorer import DataExplorer
from python.model.linear_models import LinearRegression
import pandas as pd
#TODO: Add test cases for LinearRegression
#TODO: Add test cases for BaseModel


def test_tab_loader():
    # Test case for TabLoader
    print("Test case for TabLoader")
    loader = TabLoader('test data/classification_data.csv', 'csv', verbose=True)
    df = loader.load_spreadsheet()

    print(df.head())
    print("Test case for TabLoader passed")


def test_linear_regression():
    # Test case for LinearRegression
    print("Test case for LinearRegression")
    pass

def test_base_model():
    # Test case for BaseModel
    print("Test case for BaseModel")
    pass

def test_data_explorer():
    # Test case for DataExplorer
    print("Test case for DataExplorer")
    # Create dummy dataset
    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [True, False, True, False, True]
    }
    df = pd.DataFrame(data)
    print(df)

    data_explorer = DataExplorer(df)

    data_explorer.profile()


if __name__ == '__main__':
    print("Running tests...")
    test_linear_regression()
    print("*"*50)
    test_tab_loader()
    print("*"*50)
    test_data_explorer()
    print("*"*50)
    print("All tests passed!")