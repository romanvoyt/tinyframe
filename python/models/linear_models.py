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
    """
    Logistic Regression model for binary classification.

    Parameters:
    - learning_rate (float): The learning rate for gradient descent. Default is 0.01337.
    - n_iters (int): The number of iterations for gradient descent. Default is 1337.

    Attributes:
    - learning_rate (float): The learning rate for gradient descent.
    - n_iters (int): The number of iterations for gradient descent.
    - weights (ndarray): The weights of the model.
    - bias (float): The bias term of the model.

    Methods:
    - sigmoid(x): Computes the sigmoid function.
    - fit(X, y): Fits the model to the training data.
    - predict(X): Predicts the class labels for the input data.
    - loss(y, y_predicted): Computes the loss function.

    """

    def __init__(self, learning_rate=0.01337, n_iters=1337):
        """
        Initialize the LogisticRegression model.

        Parameters:
        - learning_rate (float): The learning rate for gradient descent. Default is 0.01337.
        - n_iters (int): The number of iterations for gradient descent. Default is 1337.

        Raises:
        - ValueError: If learning_rate is less than or equal to 0.
        - ValueError: If n_iters is less than or equal to 0.

        """
        super().__init__()
        if learning_rate <= 0:
            raise ValueError('Learning rate must be greater than 0')
        if n_iters <= 0:
            raise ValueError('Number of iterations must be greater than 0')
        self.learning_rate = learning_rate
        self.n_iters = n_iters
    
    def sigmoid(self, x):
        """
        Compute the sigmoid function.

        Parameters:
        - x (ndarray): The input array.

        Returns:
        - ndarray: The output array after applying the sigmoid function.

        """
        if x.any() >= 0:
            return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        - X (ndarray): The input features.
        - y (ndarray): The target labels.

        """
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
        """
        Predict the class labels for the input data.

        Parameters:
        - X (ndarray): The input features.

        Returns:
        - ndarray: The predicted class labels.

        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def loss(self, y, y_predicted):
        """
        Compute the loss function.

        Parameters:
        - y (ndarray): The target labels.
        - y_predicted (ndarray): The predicted probabilities.

        Returns:
        - float: The computed loss.

        """
        return - (1 / len(y)) * np.sum(y * np.log(y_predicted) + (1 - y) * (np.log(1 - y_predicted)))
    


class RidgeRegression(BaseModel):
    """
    Ridge Regression model.

    Parameters:
    - alpha (float): Regularization strength.

    Methods:
    - fit(X, y): Fit the model to the training data.
    - predict(X): Make predictions on new data.

    Attributes:
    - weights (ndarray): Model weights after fitting the data.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if type(X) != np.ndarray or type(X) != np.array:
            X = np.array(X)
        if type(y) != np.array:
            y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.linalg.inv(X.T @ X + self.alpha * np.eye(n_features)) @ X.T @ y

    def predict(self, X):
        if type(X) != np.ndarray or type(X) != np.array:
            X = np.array(X)
        return X @ self.weights
    

class LassoRegression(BaseModel):
    def __init__(self, iterations, learning_rate, l1):
        super().__init__()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l1 = l1

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if type(X) != np.ndarray or type(X) != np.array:
            X = np.array(X)
        if type(y) != np.array:
            y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        for _ in range(self.iterations):
            y_predicted = X @ self.weights + self.bias
            dw = (1 / n_samples) * X.T @ (y_predicted - y) + self.l1 * np.sign(self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        if type(X) != np.ndarray or type(X) != np.array:
            X = np.array(X)
        return X @ self.weights + self.bias
