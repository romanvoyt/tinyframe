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
    def __init__(self, num_features):
        super().__init__(num_features)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predict_proba(self, features):
        if len(features) != self.num_features:
            raise ValueError("Number of features does not match the model")

        prediction = self.bias
        for i in range(self.num_features):
            prediction += self.weights[i] * features[i]

        probability = self.sigmoid(prediction)
        return probability

    def predict(self, features):
        probability = self.predict_proba(features)
        if probability >= 0.5:
            return 1
        else:
            return 0

    def fit(self, features, labels, learning_rate=0.01, num_epochs=100):
        self.train(features, labels, learning_rate, num_epochs)