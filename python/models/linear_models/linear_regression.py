import numpy as np
class BaseModel:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        raise NotImplementedError


    def predict(self, X):
        raise NotImplementedError


class LinearRegression(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.weights