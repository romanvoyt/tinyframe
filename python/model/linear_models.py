class BaseModel:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = [0.0] * num_features
        self.bias = 0.0

    def predict(self, features):
        if len(features) != self.num_features:
            raise ValueError("Number of features does not match the model")
        
        prediction = self.bias
        for i in range(self.num_features):
            prediction += self.weights[i] * features[i]
        
        return prediction

    def train(self, features, labels, learning_rate=0.01, num_epochs=100):
        if len(features) != len(labels):
            raise ValueError("Number of features and labels do not match")
        
        for epoch in range(num_epochs):
            for i in range(len(features)):
                prediction = self.predict(features[i])
                error = labels[i] - prediction
                
                self.bias += learning_rate * error
                
                for j in range(self.num_features):
                    self.weights[j] += learning_rate * error * features[i][j]