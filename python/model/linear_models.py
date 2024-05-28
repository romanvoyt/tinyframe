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


class LogisticRegression(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for i in range(1000):
            z = X @ self.weights
            h = 1 / (1 + np.exp(-z))
            gradient = X.T @ (h - y) / y.size
            self.weights -= 0.01 * gradient

    def predict(self, X):
        return X @ self.weights
    
    def predict_proba(self, X):
        return 1 / (1 + np.exp(-X @ self.weights))
    