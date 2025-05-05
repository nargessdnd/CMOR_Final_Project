import numpy as np
from collections import Counter
from Supervised_Learning.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestBase:
    """
    Base class for Random Forest algorithms.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_features : str or int, default='sqrt'
        The number of features to consider for the best split:
        - If int, then consider max_features features.
        - If 'sqrt', then max_features=sqrt(n_features).
        - If 'log2', then max_features=log2(n_features).
        - If None, then max_features=n_features.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Attributes:
    -----------
    trees : list
        The collection of fitted decision trees.
    feature_importances_ : array
        The feature importances based on impurity decrease.
    """
    
    def __init__(self, n_estimators=100, max_features='sqrt', bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        
    def _bootstrap_sample(self, X, y, rng):
        """
        Create a bootstrap sample.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.
        rng : numpy.random.RandomState
            Random number generator.
            
        Returns:
        --------
        tuple
            (X_bootstrap, y_bootstrap)
        """
        n_samples = X.shape[0]
        idxs = rng.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _get_max_features(self, n_features):
        """
        Calculate the number of features to consider for the best split.
        
        Parameters:
        -----------
        n_features : int
            Total number of features.
            
        Returns:
        --------
        int
            Number of features to consider.
        """
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif self.max_features is None:
            return n_features
        else:
            raise ValueError("max_features must be int, 'sqrt', 'log2', or None")
    
    def fit(self, X, y):
        """
        Build the random forest.
        
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
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement fit")
    
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
            Predicted values.
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement predict")
    
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
    
    def _calculate_feature_importances(self):
        """
        Calculate feature importances based on impurity decrease.
        This is a placeholder method and should be implemented by specific subclasses
        when they have trees with feature_importance_ attribute.
        """
        pass

class RandomForestClassifier(RandomForestBase):
    """
    Random Forest classifier.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : str, default='gini'
        Function to measure the quality of a split. Supported criteria: 'gini', 'entropy'.
    max_depth : int or None, default=None
        Maximum depth of the trees. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : str or int, default='sqrt'
        The number of features to consider for the best split.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt', 
                 bootstrap=True, random_state=42):
        super().__init__(n_estimators, max_features, bootstrap, random_state)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._classes = None
        
    def fit(self, X, y):
        """
        Build the random forest classifier.
        
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
        
        # Store classes
        self._classes = np.unique(y)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Calculate number of features to consider for best split
        max_features = self._get_max_features(n_features)
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Clear any existing trees
        self.trees = []
        
        # Create trees
        for i in range(self.n_estimators):
            # Set different random state for each tree
            tree_random_state = rng.randint(0, 1000000)
            
            # Create a decision tree
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_random_state
            )
            
            # Get bootstrap sample if bootstrapping is enabled
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y, rng)
            else:
                X_sample, y_sample = X, y
                
            # Fit the tree
            tree.fit(X_sample, y_sample)
            
            # Add the tree to the forest
            self.trees.append(tree)
            
        # Calculate feature importances if possible
        self._calculate_feature_importances()
            
        return self
    
    def predict(self, X):
        """
        Predict class labels for input samples.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
            
        Returns:
        --------
        array, shape = [n_samples]
            Predicted class labels.
        """
        X = np.array(X)
        
        # Make predictions with each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to get predictions by sample
        predictions = predictions.T
        
        # Take majority vote for each sample
        maj_votes = np.array([Counter(pred).most_common(1)[0][0] for pred in predictions])
        
        return maj_votes
    
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
        
        # Get predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate class probabilities for each sample
        for i in range(n_samples):
            class_counts = np.bincount(tree_preds[:, i].astype(int), minlength=len(self._classes))
            probabilities[i] = class_counts / self.n_estimators
            
        return probabilities
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
        y : array-like, shape = [n_samples]
            True labels for X.
            
        Returns:
        --------
        float
            Accuracy of the model.
        """
        return np.mean(self.predict(X) == y)
    
    def _calculate_feature_importances(self):
        """
        Calculate feature importances based on impurity decrease.
        
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature. It is also known as
        the Gini importance.
        """
        # Skip if no trees yet
        if not self.trees:
            return
        
        # Get number of features from first tree
        n_features = self.trees[0].root.feature_idx if self.trees[0].root and not self.trees[0].root.is_leaf() else 0
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Currently, our simple decision tree implementation doesn't track feature importances
        # So this remains a placeholder for now
        
        # Normalize importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

class RandomForestRegressor(RandomForestBase):
    """
    Random Forest regressor.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : str, default='mse'
        Function to measure the quality of a split. Supported criteria: 'mse'.
    max_depth : int or None, default=None
        Maximum depth of the trees. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : str or int, default='sqrt'
        The number of features to consider for the best split.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=100, criterion='mse', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt', 
                 bootstrap=True, random_state=42):
        super().__init__(n_estimators, max_features, bootstrap, random_state)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, X, y):
        """
        Build the random forest regressor.
        
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
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Calculate number of features to consider for best split
        max_features = self._get_max_features(n_features)
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Clear any existing trees
        self.trees = []
        
        # Create trees
        for i in range(self.n_estimators):
            # Set different random state for each tree
            tree_random_state = rng.randint(0, 1000000)
            
            # Create a decision tree
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_random_state
            )
            
            # Get bootstrap sample if bootstrapping is enabled
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y, rng)
            else:
                X_sample, y_sample = X, y
                
            # Fit the tree
            tree.fit(X_sample, y_sample)
            
            # Add the tree to the forest
            self.trees.append(tree)
            
        # Calculate feature importances if possible
        self._calculate_feature_importances()
            
        return self
    
    def predict(self, X):
        """
        Predict regression target for input samples.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
            
        Returns:
        --------
        array, shape = [n_samples]
            Predicted target values.
        """
        X = np.array(X)
        
        # Make predictions with each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate mean prediction for each sample
        mean_predictions = np.mean(predictions, axis=0)
        
        return mean_predictions
    
    def score(self, X, y):
        """
        Calculate R^2 score.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
        y : array-like, shape = [n_samples]
            True target values for X.
            
        Returns:
        --------
        float
            R^2 score.
        """
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
    
    def _calculate_feature_importances(self):
        """
        Calculate feature importances based on impurity decrease.
        
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        """
        # Skip if no trees yet
        if not self.trees:
            return
        
        # Get number of features from first tree
        n_features = self.trees[0].root.feature_idx if self.trees[0].root and not self.trees[0].root.is_leaf() else 0
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Currently, our simple decision tree implementation doesn't track feature importances
        # So this remains a placeholder for now
        
        # Normalize importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_) 