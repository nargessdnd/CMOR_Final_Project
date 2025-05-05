import numpy as np

class LinearRegression:
    """
    Implementation of Linear Regression using both gradient descent and normal equation.
    
    Parameters:
    -----------
    method : str, default='normal_equation'
        Method to use for optimization: 'normal_equation' or 'gradient_descent'.
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
        Cost history during training (for gradient descent).
    """
    
    def __init__(self, method='normal_equation', learning_rate=0.01, 
                 n_iterations=1000, random_state=42):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.costs = []
        
    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        
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
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Initialize weights and bias
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0
        
        if self.method == 'normal_equation':
            # Add a column of ones for the bias term
            X_b = np.c_[np.ones((n_samples, 1)), X]
            
            # Calculate weights using the normal equation
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            
            # Extract bias and weights
            self.bias = theta[0]
            self.weights = theta[1:]
            
        elif self.method == 'gradient_descent':
            # Gradient descent optimization
            for _ in range(self.n_iterations):
                # Calculate predictions
                y_pred = self._predict(X)
                
                # Calculate error
                error = y_pred - y
                
                # Calculate gradients
                dw = (1/n_samples) * np.dot(X.T, error)
                db = (1/n_samples) * np.sum(error)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Calculate cost (MSE)
                cost = (1/(2*n_samples)) * np.sum(error**2)
                self.costs.append(cost)
        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")
            
        return self
    
    def _predict(self, X):
        """
        Calculate the predicted values.
        
        Parameters:
        -----------
        X : array-like
            Input features.
            
        Returns:
        --------
        array
            Predicted values.
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Predict target values for samples in X.
        
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
        return self._predict(X)
    
    def score(self, X, y):
        """
        Calculate the coefficient of determination R^2.
        
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