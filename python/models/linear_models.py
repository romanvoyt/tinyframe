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
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if type(X) != np.ndarray or type(X) != np.array:
            X = np.array(X)
        if type(y) != np.array:
            y = np.array(y)
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        if type(X) != np.ndarray or type(X) != np.array:
            X = np.array(X)
        return X @ self.weights
    
class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=0.01337, n_iters=1337):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError('Learning rate must be greater than 0')
        if n_iters <= 0:
            raise ValueError('Number of iterations must be greater than 0')
        self.learning_rate = learning_rate
        self.n_iters = n_iters
    
    def sigmoid(self, x):
        if x.any() >= 0:
            return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if _ % 10 == 0:
                print(f'Iteration: {_}, Loss: {self.loss(y, y_predicted)}')

    def predict(self, X) -> np.array:
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def loss(self, y, y_predicted):
        return - (1 / len(y)) * np.sum(y * np.log(y_predicted) + (1 - y) * (np.log(1 - y_predicted)))