import numpy as np

class Perceptron:
    """
    Implementation of the Perceptron algorithm.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for weight updates.
    n_iterations : int, default=1000
        Number of passes over the training data.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Attributes:
    -----------
    weights : array
        Weights after fitting.
    bias : float
        Bias after fitting.
    errors : list
        Number of misclassifications in each iteration.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors = []
        
    def fit(self, X, y):
        """
        Fit the perceptron model on the training data.
        
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
        # Initialize random number generator for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        # Convert inputs to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        
        # Check if binary classification and convert labels to -1, 1
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError("Perceptron only supports binary classification.")
        
        # Map y to -1, 1
        y_mapped = np.where(y == unique_y[0], -1, 1)
        
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        
        # Training loop
        for _ in range(self.n_iterations):
            errors = 0
            
            # Update weights for each sample
            for xi, yi in zip(X, y_mapped):
                # Calculate prediction
                y_pred = self._predict_raw(xi)
                
                # Update weights if misclassified
                if yi * y_pred <= 0:  # Misclassification
                    update = self.learning_rate * yi
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            
            # Record number of misclassifications
            self.errors.append(errors)
            
            # If no errors, break early
            if errors == 0:
                break
                
        return self
    
    def _predict_raw(self, X):
        """
        Calculate raw output of the perceptron.
        
        Parameters:
        -----------
        X : array-like
            Input features.
            
        Returns:
        --------
        float
            Raw output value before thresholding.
        """
        return np.dot(X, self.weights) + self.bias
    
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
        # Apply threshold
        y_pred = np.where(self._predict_raw(X) >= 0.0, 1, 0)
        return y_pred
    
    def score(self, X, y):
        """
        Calculate accuracy score for the model.
        
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