import numpy as np

class NeuralNetwork:
    """
    Implementation of a simple feed-forward neural network with one hidden layer.
    
    Parameters:
    -----------
    hidden_layer_size : int, default=32
        Number of neurons in the hidden layer.
    activation : str, default='sigmoid'
        Activation function to use: 'sigmoid', 'relu', or 'tanh'.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of passes over the training data.
    batch_size : int, default=32
        Size of mini-batches for training.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Attributes:
    -----------
    W1 : array
        Weights from input to hidden layer.
    b1 : array
        Biases for hidden layer.
    W2 : array
        Weights from hidden to output layer.
    b2 : array
        Biases for output layer.
    losses : list
        Loss history during training.
    """
    
    def __init__(self, hidden_layer_size=32, activation='sigmoid', 
                 learning_rate=0.01, n_iterations=1000, batch_size=32, 
                 random_state=42):
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.losses = []
        
    def _activation_function(self, z):
        """
        Apply the activation function.
        
        Parameters:
        -----------
        z : array
            Input array.
            
        Returns:
        --------
        array
            Activated values.
        """
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError("Activation must be 'sigmoid', 'relu', or 'tanh'")
            
    def _activation_derivative(self, z):
        """
        Compute the derivative of the activation function.
        
        Parameters:
        -----------
        z : array
            Input array.
            
        Returns:
        --------
        array
            Derivative values.
        """
        if self.activation == 'sigmoid':
            s = self._activation_function(z)
            return s * (1 - s)
        elif self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
        else:
            raise ValueError("Activation must be 'sigmoid', 'relu', or 'tanh'")
            
    def _forward_pass(self, X):
        """
        Perform a forward pass through the network.
        
        Parameters:
        -----------
        X : array, shape = [n_samples, n_features]
            Input data.
            
        Returns:
        --------
        tuple
            (activation1, z2, activation2)
        """
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._activation_function(z1)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))  # Sigmoid for output layer
        
        return a1, z2, a2
    
    def fit(self, X, y):
        """
        Fit the neural network on the training data.
        
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
        
        # Handle multi-class classification
        if len(y.shape) == 1:
            # Binary classification
            y = y.reshape(-1, 1)
            n_outputs = 1
        else:
            # Multi-class one-hot encoded
            n_outputs = y.shape[1]
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Initialize weights and biases
        self.W1 = rng.normal(0, 0.1, size=(n_features, self.hidden_layer_size))
        self.b1 = np.zeros(self.hidden_layer_size)
        self.W2 = rng.normal(0, 0.1, size=(self.hidden_layer_size, n_outputs))
        self.b2 = np.zeros(n_outputs)
        
        # Mini-batch gradient descent
        for _ in range(self.n_iterations):
            # Shuffle the data
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Create mini-batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                batch_size = X_batch.shape[0]
                
                # Forward pass
                a1, z2, a2 = self._forward_pass(X_batch)
                
                # Compute loss
                epsilon = 1e-15
                loss = -np.mean(
                    y_batch * np.log(a2 + epsilon) + 
                    (1 - y_batch) * np.log(1 - a2 + epsilon)
                )
                
                # Backpropagation
                dz2 = a2 - y_batch
                dW2 = (1/batch_size) * np.dot(a1.T, dz2)
                db2 = (1/batch_size) * np.sum(dz2, axis=0)
                
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self._activation_derivative(np.dot(X_batch, self.W1) + self.b1)
                dW1 = (1/batch_size) * np.dot(X_batch.T, dz1)
                db1 = (1/batch_size) * np.sum(dz1, axis=0)
                
                # Update parameters
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
            
            # Record loss after each epoch
            _, _, a2 = self._forward_pass(X)
            epsilon = 1e-15
            loss = -np.mean(
                y * np.log(a2 + epsilon) + 
                (1 - y) * np.log(1 - a2 + epsilon)
            )
            self.losses.append(loss)
            
        return self
    
    def predict_proba(self, X):
        """
        Calculate probability estimates.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input samples.
            
        Returns:
        --------
        array, shape = [n_samples, n_outputs]
            Probability estimates.
        """
        X = np.array(X)
        _, _, a2 = self._forward_pass(X)
        return a2
    
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
        if probas.shape[1] == 1:
            # Binary classification
            return (probas >= threshold).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(probas, axis=1)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            True values.
            
        Returns:
        --------
        float
            Accuracy of the model.
        """
        X = np.array(X)
        y = np.array(y)
        
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Multi-class (one-hot encoded)
            y_true = np.argmax(y, axis=1)
        else:
            # Binary or multi-class (1D array)
            y_true = y.flatten()
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true) 