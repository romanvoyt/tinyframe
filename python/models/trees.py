import numpy as np

class TreeNode:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

class DecisionTree:
    def __init__(self, max_depth=None) -> None:
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        return self._grow_tree(X, y, depth=0)
    
    def _grow_tree(self, X, y, depth):
        node = TreeNode(max_depth=depth)
        if self._stopping_criteria:
            node.value = self._most_common_label(y)
        
        best_split = self.find_best_split(X,y)
        if best_split is None:
            node.value = self._most_common_label(y)
            return node
        
        feature_idx, threshold = best_split
        node.feature_index = feature_idx
        node.threshold = threshold

        left_indices = X[:, feature_idx] < threshold
        right_indices = ~left_indices

        node.left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        node.right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return node
            
    
    def _stopping_criteria(self, y, depth):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth > self.max_depth):
            return True
        return False
    
    def find_best_split(self, X, y):
        best_gini = 1
        best_split = None

        for feature_index in range(X.shape[1]):
            tresholds = np.unique(X[:, feature_index])
            for treshhold in tresholds:
                left_ids = X[:, feature_index] < treshhold
                gini = self._gini_impurity(y[left_ids], y[~left_ids])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, treshhold)
        
        return best_split


    def _gini(self, left_y, right_y):
        prob_left = len(left_y) / (len(left_y) + len(right_y))
        prob_right = 1 - prob_left

        gini_left = 1 - sum((np.bincount(left_y)/len(left_y)) ** 2)
        gini_right = 1 - sum((np.bincount(right_y) / len(right_y)) ** 2)

        gini = prob_left * gini_left + prob_right * gini_right
        return gini

    
    def _most_common_label(self, y):
        # Example: Determine most common label
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
