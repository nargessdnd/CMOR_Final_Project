import numpy as np
from Supervised_Learning.decision_tree import DecisionTreeClassifier

class AdaBoostClassifier:
    """
    AdaBoost classifier based on SAMME algorithm.
    
    Parameters:
    -----------
    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If None, a decision tree with max_depth=1 (decision stump) is used.
    n_estimators : int, default=50
        Maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Attributes:
    -----------
    estimators_ : list
        The collection of fitted base estimators.
    estimator_weights_ : array
        Weights for each estimator in the boosted ensemble.
    estimator_errors_ : array
        Classification error for each estimator in the boosted ensemble.
    classes_ : array
        Unique class labels.
    """
    
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(n_estimators)
        self.estimator_errors_ = np.ones(n_estimators)
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Build a boosted classifier from the training data.
        
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
        
        # Get the unique class labels
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Check if binary classification
        if n_classes != 2:
            raise ValueError("AdaBoostClassifier currently only supports binary classification.")
        
        # Map y values to -1, 1 for easier calculations
        y_values = np.ones(y.shape)
        y_values[y == self.classes_[0]] = -1
        
        # Initialize weights for samples
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Create and train the weak classifiers
        for i in range(self.n_estimators):
            # Create decision tree stump if no base_estimator is provided
            if self.base_estimator is None:
                estimator = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(0, 1000000))
            else:
                # Clone base_estimator (here we just create a new instance)
                estimator = type(self.base_estimator)(**self.base_estimator.get_params())
            
            # Fit the estimator with sample weights
            # Note: our DecisionTreeClassifier doesn't support sample weights
            # So we use a simple weighted boostrap sample as a workaround
            bootstrap_idxs = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights / np.sum(sample_weights)
            )
            X_bootstrap = X[bootstrap_idxs]
            y_bootstrap = y[bootstrap_idxs]
            
            # Train the estimator
            estimator.fit(X_bootstrap, y_bootstrap)
            
            # Predict with the estimator
            y_pred = estimator.predict(X)
            
            # Map predictions to -1, 1
            y_pred_values = np.ones(y_pred.shape)
            y_pred_values[y_pred == self.classes_[0]] = -1
            
            # Calculate the weighted error
            incorrect = y_pred_values != y_values
            estimator_error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # If the error is too large, break
            if estimator_error >= 1.0 - 1/n_classes:
                break
                
            # Calculate estimator weight
            estimator_weight = self.learning_rate * np.log((1 - estimator_error) / max(estimator_error, 1e-10))
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * incorrect)
            
            # Normalize sample weights
            sample_weights /= np.sum(sample_weights)
            
            # Save the estimator and its weight
            self.estimators_.append(estimator)
            self.estimator_weights_[i] = estimator_weight
            self.estimator_errors_[i] = estimator_error
            
            # If perfect fit, break
            if estimator_error == 0:
                break
                
        # Trim estimator arrays to actual estimators
        n_estimators = len(self.estimators_)
        self.estimator_weights_ = self.estimator_weights_[:n_estimators]
        self.estimator_errors_ = self.estimator_errors_[:n_estimators]
            
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
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
        
        # Check if estimators
        if not self.estimators_:
            raise ValueError("Estimators not fitted. Call fit before predict.")
            
        # Compute predictions for each estimator
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # For binary classification, use simple weighted majority voting
        y_pred = np.zeros(n_samples)
        
        for i, estimator in enumerate(self.estimators_):
            # Get predictions from the estimator
            estimator_pred = estimator.predict(X)
            
            # Map predictions to -1, 1
            estimator_pred_values = np.ones(estimator_pred.shape)
            estimator_pred_values[estimator_pred == self.classes_[0]] = -1
            
            # Add weighted contribution to the final prediction
            y_pred += self.estimator_weights_[i] * estimator_pred_values
            
        # Convert predictions back to class labels
        return np.where(y_pred >= 0, self.classes_[1], self.classes_[0])
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
            
        Returns:
        --------
        array, shape = [n_samples, n_classes]
            Probability estimates.
        """
        X = np.array(X)
        
        # Check if estimators exist
        if not self.estimators_:
            raise ValueError("Estimators not fitted. Call fit before predict_proba.")
            
        # Compute predictions for each estimator
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Calculate the sum of weights
        total_weight = np.sum(self.estimator_weights_)
        
        # Calculate weighted score as probabilities
        y_pred = np.zeros(n_samples)
        
        for i, estimator in enumerate(self.estimators_):
            # Get predictions from the estimator
            estimator_pred = estimator.predict(X)
            
            # Map predictions to -1, 1
            estimator_pred_values = np.ones(estimator_pred.shape)
            estimator_pred_values[estimator_pred == self.classes_[0]] = -1
            
            # Add weighted contribution to the final prediction
            y_pred += self.estimator_weights_[i] * estimator_pred_values
        
        # Convert scores to probabilities in [0, 1]
        probas = np.zeros((n_samples, n_classes))
        
        # Convert to raw probability
        proba_positive = 1.0 / (1.0 + np.exp(-2 * y_pred))
        probas[:, 0] = 1.0 - proba_positive
        probas[:, 1] = proba_positive
            
        return probas
    
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

class GradientBoostingRegressor:
    """
    Gradient Boosting for regression.
    
    This algorithm builds an additive model in a forward stage-wise manner.
    It allows for the optimization of arbitrary differentiable loss functions.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual regression trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Attributes:
    -----------
    estimators_ : list
        The collection of fitted base estimators.
    init_pred_ : float
        The initial prediction (usually mean of target values).
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.estimators_ = []
        self.init_pred_ = None
        
    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        
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
        from Supervised_Learning.decision_tree import DecisionTreeRegressor
        
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Initialize with the mean of the target
        self.init_pred_ = np.mean(y)
        
        # Initialize predictions with the mean
        y_pred = np.full(y.shape, self.init_pred_)
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Iterate through the number of estimators
        for i in range(self.n_estimators):
            # Calculate the residuals (negative gradients for MSE loss)
            residuals = y - y_pred
            
            # Create a regression tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 1000000)
            )
            
            # Fit the tree to the residuals
            tree.fit(X, residuals)
            
            # Update the predictions
            update = self.learning_rate * tree.predict(X)
            y_pred += update
            
            # Save the estimator
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Predict regression target for X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
            
        Returns:
        --------
        array, shape = [n_samples]
            Predicted values.
        """
        X = np.array(X)
        
        # Start with the initial prediction
        y_pred = np.full(X.shape[0], self.init_pred_)
        
        # Add the contribution of each tree
        for i, tree in enumerate(self.estimators_):
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
    
    def score(self, X, y):
        """
        Calculate R^2 score.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
        y : array-like, shape = [n_samples]
            True values for X.
            
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