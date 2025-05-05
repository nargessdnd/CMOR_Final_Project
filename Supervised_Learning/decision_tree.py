import numpy as np
from collections import Counter

class Node:
    """
    Node class for Decision Tree.
    
    Parameters:
    -----------
    feature_idx : int or None
        Index of the feature to split on. None for leaf nodes.
    threshold : float or None
        Threshold value for the feature split. None for leaf nodes.
    left : Node or None
        Left child node.
    right : Node or None
        Right child node.
    value : object
        Value for leaf nodes (prediction value).
    """
    
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        """Check if the node is a leaf node."""
        return self.value is not None

class DecisionTree:
    """
    Base class for Decision Tree algorithms.
    
    Parameters:
    -----------
    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.root = None
        
    def fit(self, X, y):
        """
        Build the decision tree.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns:
        --------
        self : object
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Build the tree
        self.root = self._grow_tree(X, y, depth=0, rng=rng)
        
        return self
    
    def _grow_tree(self, X, y, depth, rng):
        """
        Recursively grow the decision tree.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.
        depth : int
            Current depth of the tree.
        rng : numpy.random.RandomState
            Random number generator.
            
        Returns:
        --------
        Node
            Root node of the subtree.
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            self._is_pure(y)):
            # Create a leaf node
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Find the best split
        feature_idxs = np.arange(n_features)
        
        best_feature_idx, best_threshold = self._best_split(X, y, feature_idxs, rng)
        
        # Create child nodes
        left_idxs = X[:, best_feature_idx] <= best_threshold
        right_idxs = ~left_idxs
        
        # Check if the split is valid
        if (np.sum(left_idxs) < self.min_samples_leaf or
            np.sum(right_idxs) < self.min_samples_leaf):
            # Create a leaf node
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Recursively grow the left and right subtrees
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1, rng)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1, rng)
        
        return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left, right=right)
    
    def predict(self, X):
        """
        Predict class labels or target values for input samples.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
            
        Returns:
        --------
        array, shape = [n_samples]
            Predicted class labels or target values.
        """
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse the tree and make a prediction for a single sample.
        
        Parameters:
        -----------
        x : array-like, shape = [n_features]
            Input sample.
        node : Node
            Current node in the tree.
            
        Returns:
        --------
        object
            Predicted value.
        """
        if node.is_leaf():
            return node.value
            
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def _best_split(self, X, y, feature_idxs, rng):
        """
        Find the best feature and threshold for splitting.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.
        feature_idxs : array-like
            Indices of features to consider for the split.
        rng : numpy.random.RandomState
            Random number generator.
            
        Returns:
        --------
        tuple
            (best_feature_idx, best_threshold)
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _best_split")
    
    def _calculate_leaf_value(self, y):
        """
        Calculate the value for a leaf node.
        
        Parameters:
        -----------
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns:
        --------
        object
            Leaf value.
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _calculate_leaf_value")
    
    def _is_pure(self, y):
        """
        Check if a node is pure.
        
        Parameters:
        -----------
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns:
        --------
        bool
            True if the node is pure (all samples have the same target value).
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _is_pure")
    
    def score(self, X, y):
        """
        Calculate score for the model.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
        y : array-like, shape = [n_samples]
            True values for X.
            
        Returns:
        --------
        float
            Score of the model.
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement score")

class DecisionTreeClassifier(DecisionTree):
    """
    Decision Tree classifier.
    
    Parameters:
    -----------
    criterion : str, default='gini'
        Function to measure the quality of a split. Supported criteria: 'gini', 'entropy'.
    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, random_state)
        self.criterion = criterion
        self._classes = None
    
    def fit(self, X, y):
        """
        Build the decision tree classifier.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns:
        --------
        self : object
        """
        # Store classes
        self._classes = np.unique(y)
        
        return super().fit(X, y)
    
    def _is_pure(self, y):
        """Check if a node is pure (all samples have the same class)."""
        return len(np.unique(y)) == 1
    
    def _calculate_leaf_value(self, y):
        """Return the most common class in a leaf node."""
        return Counter(y).most_common(1)[0][0]
    
    def _best_split(self, X, y, feature_idxs, rng):
        """Find the best feature and threshold for splitting."""
        best_gain = -float('inf')
        best_feature_idx = feature_idxs[0]  # Default to first feature
        best_threshold = 0
        
        # Calculate current impurity
        current_impurity = self._calculate_impurity(y)
        
        # Try each feature
        for feature_idx in feature_idxs:
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Skip if only one unique value
            if len(thresholds) == 1:
                continue
            
            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = ~left_idxs
                
                # Skip if split doesn't meet min_samples_leaf
                if (np.sum(left_idxs) < self.min_samples_leaf or
                    np.sum(right_idxs) < self.min_samples_leaf):
                    continue
                
                # Calculate impurity for the split
                left_impurity = self._calculate_impurity(y[left_idxs])
                right_impurity = self._calculate_impurity(y[right_idxs])
                
                # Calculate the weighted average impurity
                n = len(y)
                n_left = np.sum(left_idxs)
                n_right = np.sum(right_idxs)
                
                weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
                
                # Calculate information gain
                gain = current_impurity - weighted_impurity
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on the criterion."""
        # Handle empty array
        if len(y) == 0:
            return 0
            
        # Calculate class probabilities
        class_counts = np.bincount(y.astype(int))
        probabilities = class_counts / len(y)
        
        if self.criterion == 'gini':
            # Gini impurity
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            # Entropy
            # Avoid log(0) by filtering out zero probabilities
            nonzero_probs = probabilities[probabilities > 0]
            return -np.sum(nonzero_probs * np.log2(nonzero_probs))
        else:
            raise ValueError("Supported criteria are 'gini' and 'entropy'")
    
    def score(self, X, y):
        """Calculate accuracy score."""
        return np.mean(self.predict(X) == y)
    
    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
            
        Returns:
        --------
        array, shape = [n_samples, n_classes]
            Probability estimates.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self._classes)
        
        # Initialize probabilities
        probabilities = np.zeros((n_samples, n_classes))
        
        # Calculate probabilities for each sample
        for i, x in enumerate(X):
            # Traverse tree to find the leaf node
            node = self.root
            while not node.is_leaf():
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            
            # Set probability as 1.0 for the predicted class
            predicted_class = node.value
            class_idx = np.where(self._classes == predicted_class)[0][0]
            probabilities[i, class_idx] = 1.0
                
        return probabilities

class DecisionTreeRegressor(DecisionTree):
    """
    Decision Tree regressor.
    
    Parameters:
    -----------
    criterion : str, default='mse'
        Function to measure the quality of a split. Supported criteria: 'mse'.
    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, random_state)
        self.criterion = criterion
    
    def _is_pure(self, y):
        """Check if a node is pure (all samples have the same target value)."""
        return len(np.unique(y)) == 1
    
    def _calculate_leaf_value(self, y):
        """Return the mean of target values in a leaf node."""
        return np.mean(y)
    
    def _best_split(self, X, y, feature_idxs, rng):
        """Find the best feature and threshold for splitting."""
        best_error = float('inf')
        best_feature_idx = feature_idxs[0]  # Default to first feature
        best_threshold = 0
        
        # Try each feature
        for feature_idx in feature_idxs:
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Skip if only one unique value
            if len(thresholds) == 1:
                continue
            
            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = ~left_idxs
                
                # Skip if split doesn't meet min_samples_leaf
                if (np.sum(left_idxs) < self.min_samples_leaf or
                    np.sum(right_idxs) < self.min_samples_leaf):
                    continue
                
                # Calculate error for the split
                left_error = self._calculate_error(y[left_idxs])
                right_error = self._calculate_error(y[right_idxs])
                
                # Calculate the weighted average error
                n = len(y)
                n_left = np.sum(left_idxs)
                n_right = np.sum(right_idxs)
                
                weighted_error = (n_left / n) * left_error + (n_right / n) * right_error
                
                # Update best split if this is better
                if weighted_error < best_error:
                    best_error = weighted_error
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def _calculate_error(self, y):
        """Calculate error based on the criterion."""
        # Handle empty array
        if len(y) == 0:
            return 0
            
        if self.criterion == 'mse':
            # Mean squared error
            mean = np.mean(y)
            return np.mean((y - mean) ** 2)
        else:
            raise ValueError("Supported criterion is 'mse'")
    
    def score(self, X, y):
        """Calculate R^2 score."""
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        
        # Calculate R^2 score
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        
        # Avoid division by zero
        if v == 0:
            return 0.0
            
        return 1 - (u / v) 