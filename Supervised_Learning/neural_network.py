# Supervised_Learning/neural_network.py

import numpy as np
import pandas as pd

class NeuralNetwork:
    """
    A simple Feedforward Neural Network implementation with one hidden layer.
    Can be used for binary classification or regression.

    Parameters
    ----------
    hidden_layer_size : int, optional (default=100)
        Number of neurons in the hidden layer.
    task_type : str, optional (default='classification')
        Type of task: 'classification' (binary) or 'regression'.
    activation : str, optional (default='relu')
        Activation function for the hidden layer ('relu' or 'sigmoid').
    learning_rate : float, optional (default=0.01)
        Step size for weight updates during gradient descent.
    n_iterations : int, optional (default=1000)
        Number of training iterations (epochs).
    batch_size : int, optional (default=32)
        Number of samples per batch for mini-batch gradient descent.
        If None or >= n_samples, standard gradient descent is used.
    random_state : int, optional (default=None)
        Seed for random number generation for weight initialization and batch shuffling.
    verbose : bool, optional (default=False)
        Whether to print loss during training.
    print_every : int, optional (default=100)
        How often to print loss if verbose is True.
    """
    def __init__(self, hidden_layer_size=100, task_type='classification', activation='relu',
                 learning_rate=0.01, n_iterations=1000, batch_size=32,
                 random_state=None, verbose=False, print_every=100):

        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")
        if activation not in ['relu', 'sigmoid']:
             raise ValueError("activation must be 'relu' or 'sigmoid'")

        self.hidden_layer_size = hidden_layer_size
        self.task_type = task_type
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.print_every = print_every

        self.weights1 = None  # Input to Hidden
        self.bias1 = None     # Hidden layer bias
        self.weights2 = None  # Hidden to Output
        self.bias2 = None     # Output layer bias
        self.losses = []      # To store loss per iteration/epoch

        self._rng = np.random.default_rng(self.random_state)

        # Set activation functions based on selection
        if self.activation_name == 'relu':
            self.activation_func = self._relu
            self.activation_derivative = self._relu_derivative
        elif self.activation_name == 'sigmoid':
            self.activation_func = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative


    def _sigmoid(self, x):
        # Clip values to avoid overflow in exp
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _sigmoid_derivative(self, x):
        sig_x = self._sigmoid(x)
        return sig_x * (1 - sig_x)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _linear(self, x):
        """Linear activation (identity) for regression output."""
        return x

    def _linear_derivative(self, x):
        """Derivative of linear activation is 1."""
        return np.ones_like(x)

    def _mse_loss(self, y_true, y_pred):
        """Mean Squared Error loss."""
        return np.mean((y_true - y_pred)**2)

    def _mse_loss_derivative(self, y_true, y_pred):
        """Derivative of MSE loss."""
        return 2 * (y_pred - y_true) / y_true.shape[0] # Average over samples

    def _binary_cross_entropy(self, y_true, y_pred):
        """Binary Cross-Entropy loss."""
        # Clip predictions to avoid log(0)
        epsilon = 1e-10
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    def _binary_cross_entropy_derivative(self, y_true, y_pred):
        """Derivative of Binary Cross-Entropy loss w.r.t. pre-activation output (z2)."""
        # Clip predictions to avoid division by zero
        epsilon = 1e-10
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        # This derivative is simpler: (y_pred - y_true) / N
        # Assumes y_pred is the output *after* sigmoid activation
        return (y_pred_clipped - y_true) / y_true.shape[0]

    def _initialize_weights(self, n_features, n_output_neurons):
        # He initialization for ReLU, Xavier/Glorot for Sigmoid
        if self.activation_name == 'relu':
            limit1 = np.sqrt(2. / n_features)
            limit2 = np.sqrt(2. / self.hidden_layer_size)
        else: # Sigmoid
            limit1 = np.sqrt(1. / n_features)
            limit2 = np.sqrt(1. / self.hidden_layer_size)

        self.weights1 = self._rng.normal(scale=limit1, size=(n_features, self.hidden_layer_size))
        self.bias1 = np.zeros((1, self.hidden_layer_size))
        self.weights2 = self._rng.normal(scale=limit2, size=(self.hidden_layer_size, n_output_neurons))
        self.bias2 = np.zeros((1, n_output_neurons))


    def _forward_pass(self, X):
        # Input to Hidden Layer
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self.activation_func(z1) # Hidden layer activation

        # Hidden to Output Layer
        z2 = np.dot(a1, self.weights2) + self.bias2

        # Output activation depends on task type
        if self.task_type == 'classification':
            a2 = self._sigmoid(z2) # Sigmoid for binary classification probability
        else: # Regression
            a2 = self._linear(z2) # Linear activation for regression output

        return z1, a1, z2, a2

    def _backward_pass(self, X, y, z1, a1, z2, a2):
        m = X.shape[0] # Number of samples in batch

        # Calculate loss and its derivative based on task type
        if self.task_type == 'classification':
            loss = self._binary_cross_entropy(y, a2)
            # Derivative of loss w.r.t z2 (output layer pre-activation)
            # For BCE with sigmoid output, this simplifies nicely
            dz2 = (a2 - y) / m
        else: # Regression
            loss = self._mse_loss(y, a2)
            # Derivative of loss w.r.t a2 (output activation)
            da2 = 2 * (a2 - y) / m
            # Derivative of loss w.r.t z2 (using chain rule: da2 * dz2/da2)
            # Since output activation is linear (derivative=1), dz2 = da2 * 1
            dz2 = da2 * self._linear_derivative(z2) # Simplified: dz2 = da2

        # Gradients for output layer weights and bias
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Gradients for hidden layer
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self.activation_derivative(z1) # Element-wise product

        # Gradients for hidden layer weights and bias
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dw1, db1, dw2, db2, loss

    def fit(self, X, y):
        """
        Train the neural network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values. Should be {0, 1} for classification.
        """
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        if y.ndim == 1: y = y.reshape(-1, 1) # Ensure y is a column vector

        n_samples, n_features = X.shape

        # Determine output layer size
        n_output_neurons = 1 # For binary classification or regression

        self._initialize_weights(n_features, n_output_neurons)

        # Determine effective batch size
        if self.batch_size is None or self.batch_size >= n_samples:
            effective_batch_size = n_samples
        else:
            effective_batch_size = self.batch_size

        n_batches = n_samples // effective_batch_size

        self.losses = []
        for i in range(self.n_iterations):
            epoch_loss = 0
            # Shuffle data each epoch for mini-batch GD
            indices = np.arange(n_samples)
            self._rng.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for batch_num in range(n_batches):
                start_idx = batch_num * effective_batch_size
                end_idx = start_idx + effective_batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                z1, a1, z2, a2 = self._forward_pass(X_batch)

                # Backward pass
                dw1, db1, dw2, db2, batch_loss = self._backward_pass(X_batch, y_batch, z1, a1, z2, a2)
                epoch_loss += batch_loss

                # Update weights and biases
                self.weights1 -= self.learning_rate * dw1
                self.bias1 -= self.learning_rate * db1
                self.weights2 -= self.learning_rate * dw2
                self.bias2 -= self.learning_rate * db2

            # Handle potential last partial batch
            if n_samples % effective_batch_size != 0:
                X_batch = X_shuffled[n_batches * effective_batch_size:]
                y_batch = y_shuffled[n_batches * effective_batch_size:]
                if X_batch.shape[0] > 0: # Check if there is a partial batch
                    z1, a1, z2, a2 = self._forward_pass(X_batch)
                    dw1, db1, dw2, db2, batch_loss = self._backward_pass(X_batch, y_batch, z1, a1, z2, a2)
                    epoch_loss += batch_loss
                    self.weights1 -= self.learning_rate * dw1
                    self.bias1 -= self.learning_rate * db1
                    self.weights2 -= self.learning_rate * dw2
                    self.bias2 -= self.learning_rate * db2


            avg_epoch_loss = epoch_loss / (n_batches + (1 if n_samples % effective_batch_size != 0 else 0))
            self.losses.append(avg_epoch_loss)

            if self.verbose and (i + 1) % self.print_every == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {avg_epoch_loss:.6f}")


    def predict_proba(self, X):
        """Predict class probabilities (only for classification task)."""
        if self.task_type != 'classification':
             raise AttributeError("predict_proba is only available for task_type='classification'")
        if isinstance(X, pd.DataFrame): X = X.values
        _, _, _, a2 = self._forward_pass(X)
        # a2 contains the probability of the positive class (class 1)
        # Return probabilities for both classes [P(class 0), P(class 1)]
        # return np.hstack((1 - a2, a2)) # If multi-class needed later
        return a2 # For binary, return P(class 1)

    def predict(self, X):
        """
        Predict class labels or regression values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels ({0, 1}) or regression values.
        """
        if isinstance(X, pd.DataFrame): X = X.values
        _, _, _, a2 = self._forward_pass(X)

        if self.task_type == 'classification':
            # Predict class 1 if probability > 0.5
            predictions = (a2 > 0.5).astype(int)
        else: # Regression
            predictions = a2 # Output is the predicted value

        return predictions.flatten() # Return as a 1D array