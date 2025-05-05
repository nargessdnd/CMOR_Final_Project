import numpy as np

class LogisticRegression:
    """
    Implementation of Logistic Regression using gradient descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of iterations for gradient descent.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Attributes:
    -----------
    weights : array
        Weights after fitting.
    bias : float
        Bias after fitting.
    costs : list
        Loss history during training.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.costs = []
        
    def _sigmoid(self, z):
        """
        Compute the sigmoid function.
        
        Parameters:
        -----------
        z : array
            Linear output.
            
        Returns:
        --------
        array
            Sigmoid of z.
        """
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        """
        Fit the logistic regression model on the training data.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values (0 or 1).
            
        Returns:
        --------
        self : object
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Initialize weights and bias
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0
        
        # Gradient descent optimization
        for _ in range(self.n_iterations):
            # Linear combination
            linear_output = np.dot(X, self.weights) + self.bias
            
            # Sigmoid activation
            y_pred = self._sigmoid(linear_output)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cost (binary cross-entropy)
            # Avoid log(0) by adding a small epsilon
            epsilon = 1e-15
            cost = -(1/n_samples) * np.sum(
                y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)
            )
            self.costs.append(cost)
            
        return self
    
    def predict_proba(self, X):
        """
        Calculate the probability estimates.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
            
        Returns:
        --------
        array, shape = [n_samples]
            Probability estimates.
        """
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
        threshold : float, default=0.5
            Threshold for binary classification.
            
        Returns:
        --------
        array, shape = [n_samples]
            Predicted class labels.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
        y : array-like, shape = [n_samples]
            True values for X.
            
        Returns:
        --------
        float
            Accuracy of the model.
        """
        return np.mean(self.predict(X) == y) 