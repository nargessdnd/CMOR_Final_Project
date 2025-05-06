# Supervised_Learning/ensemble_methods.py

import numpy as np
from sklearn.base import clone # Use clone for sklearn compatibility if needed later
#from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier # For default if none provided # Commented out if not used
from Supervised_Learning.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor # Assuming these are in the project

class AdaBoostClassifier:
    """
    AdaBoost Classifier Implementation.

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier(max_depth=1))
        The base estimator from which the boosted ensemble is built.
        Needs to support sample weights (or simulate it via weighted bootstrap).
        If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, optional (default=1.0)
        Learning rate shrinks the contribution of each classifier.
    random_state : int, optional (default=None)
        Controls the random seed given to the base estimator at each boosting iteration.
    """
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        # Use a simple decision stump from our implementation if None is provided
        if base_estimator is None:
            # Ensure the default base estimator is initialized correctly
            self.base_estimator_ = DecisionTreeClassifier(max_depth=1)
        else:
            # Store the original base estimator configuration
            self.base_estimator_ = base_estimator

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)
        self.classes_ = None


    def _get_estimator_params(self, estimator):
        """
        Helper function to get parameters from our custom estimators
        since they don't have get_params(). Add params as needed.
        Assumes the estimator instance passed holds the desired config.
        """
        params = {}
        # Check attributes directly on the instance provided
        if hasattr(estimator, 'max_depth'):
            params['max_depth'] = estimator.max_depth
        if hasattr(estimator, 'min_samples_split'):
            params['min_samples_split'] = estimator.min_samples_split
        if hasattr(estimator, 'min_samples_leaf'):
             params['min_samples_leaf'] = estimator.min_samples_leaf # Example if your DT has this
        if hasattr(estimator, 'criterion'):
            params['criterion'] = estimator.criterion
        # Add other relevant parameters from your DecisionTreeClassifier
        return params

    def fit(self, X, y):
        """
        Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Can be numpy array or pandas DataFrame.
        y : array-like of shape (n_samples,)
            The target values. Assumes binary {0, 1} or {-1, 1}.
            If {0, 1}, will be converted internally to {-1, 1} for calculations.
        """
        # Convert X to numpy array if it's a pandas DataFrame for consistent indexing
        if hasattr(X, 'iloc'):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        # Convert y to numpy array for consistent indexing and manipulation
        y_np = np.asarray(y)

        n_samples, n_features = X_np.shape
        rng = np.random.default_rng(self.random_state)

        # Store original classes and create internal {-1, 1} representation
        self.classes_ = np.unique(y_np)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoost requires binary classification problems.")

        # Create y_internal consistently as {-1, 1}
        # Map the first class found (e.g., 0) to -1 and the second (e.g., 1) to 1
        y_internal = np.where(y_np == self.classes_[0], -1, 1)


        # Initialize weights
        sample_weights = np.full(n_samples, (1 / n_samples))

        self.estimators_ = [] # Reset estimators list
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)


        for iboost in range(self.n_estimators):
            # Create a new instance of the base estimator for this iteration
            estimator_params = self._get_estimator_params(self.base_estimator_)
            estimator_type = type(self.base_estimator_)

            # Add random_state for this specific estimator if base supports it
            # Check if the *type* of estimator likely accepts random_state
            # This is heuristic; ideally check inspect.signature if it gets complex
            try:
                 # Attempt to add random_state if the base class likely supports it
                 estimator_params['random_state'] = rng.integers(0, 1000000)
                 estimator = estimator_type(**estimator_params)
            except TypeError: # If random_state is not an accepted argument
                 if 'random_state' in estimator_params: del estimator_params['random_state']
                 estimator = estimator_type(**estimator_params)


            # Fit the estimator using weighted bootstrap sampling
            # (since our DT doesn't directly support sample_weight)
            sum_weights = np.sum(sample_weights)
            if sum_weights <= 0: # Avoid division by zero if all weights become zero
                print(f"Warning: Sum of sample weights is {sum_weights} at iteration {iboost}. Stopping early.")
                break
            normalized_weights = sample_weights / sum_weights

            bootstrap_idxs = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=normalized_weights
            )

            # --- FIX: Use original y labels for fitting the tree ---
            X_bootstrap = X_np[bootstrap_idxs]
            y_bootstrap = y_np[bootstrap_idxs] # Use original labels (0, 1 etc.)

            # Fit the weak learner on the bootstrap sample with original labels
            try:
                estimator.fit(X_bootstrap, y_bootstrap)
            except Exception as e:
                print(f"Error fitting base estimator at iteration {iboost} with params {estimator_params}")
                print(f"X_bootstrap shape: {X_bootstrap.shape}, y_bootstrap unique values: {np.unique(y_bootstrap)}")
                # Optionally, re-raise the error or break
                raise e # Re-raise to see the full error from the estimator

            # --- Predict on the *original* full dataset X ---
            # The base estimator's predict should return its native labels (e.g., 0, 1)
            y_pred_orig_labels = estimator.predict(X_np)

            # --- Convert prediction to {-1, 1} for AdaBoost calculations ---
            y_pred_internal = np.where(y_pred_orig_labels == self.classes_[0], -1, 1)


            # Calculate estimator error using y_internal and sample_weights
            # Note: Use y_internal here for correct error calculation based on {-1, 1}
            incorrect = (y_pred_internal != y_internal)
            estimator_error = np.dot(sample_weights, incorrect) / sum_weights

            # Avoid division by zero and handle perfect/bad classifiers
            eps = 1e-10 # Small epsilon for numerical stability
            if estimator_error <= eps:
                # Perfect fit or close enough
                estimator_weight = 1.0 # Assign maximum weight conceptually
                self.estimators_.append(estimator)
                self.estimator_weights_[iboost] = estimator_weight
                self.estimator_errors_[iboost] = estimator_error
                # Don't break immediately, one perfect estimator is fine, let others contribute
                # If needed, break after storing: break
            elif estimator_error >= 0.5 - eps:
                 # Worse than or equal to random guessing
                 # AdaBoost theory breaks down here. Stop adding estimators.
                 # Don't add this estimator. We can stop entirely.
                 if len(self.estimators_) == 0:
                      # If the *first* estimator is bad, we can't proceed.
                      # Add it with zero weight or raise error. Let's add with 0 weight.
                      self.estimators_.append(estimator)
                      self.estimator_weights_[iboost] = 0.0
                      self.estimator_errors_[iboost] = estimator_error
                      print(f"Warning: First estimator error >= 0.5 ({estimator_error:.4f}). Ensemble may not work well.")
                      # Optionally break here if no estimators should be kept
                 else:
                      print(f"Estimator error >= 0.5 ({estimator_error:.4f}) at iteration {iboost}. Stopping boosting.")
                 break # Stop boosting process

            # Calculate estimator weight (alpha) if error is valid (0 < error < 0.5)
            estimator_weight = self.learning_rate * 0.5 * np.log((1.0 - estimator_error) / (estimator_error + eps))

            # Store the estimator and its weight
            self.estimators_.append(estimator)
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Update sample weights using y_internal and y_pred_internal
            sample_weights *= np.exp(-estimator_weight * y_internal * y_pred_internal)
            # Normalize sample weights for the next iteration
            sample_weights /= np.sum(sample_weights)


        # Trim unused estimator slots if we stopped early
        actual_n_estimators = len(self.estimators_)
        if actual_n_estimators == 0:
             print("Warning: No valid estimators were trained. AdaBoost model might not predict correctly.")
             # Handle this case in predict: perhaps predict the majority class or raise error
        self.estimator_weights_ = self.estimator_weights_[:actual_n_estimators]
        self.estimator_errors_ = self.estimator_errors_[:actual_n_estimators]

        return self

    def predict(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Can be numpy array or pandas DataFrame.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted classes in the original format (e.g., {0, 1}).
        """
        if not self.estimators_:
             raise RuntimeError("AdaBoostClassifier has not been fitted yet or no valid estimators were trained.")

        # Convert X to numpy array if it's a pandas DataFrame
        if hasattr(X, 'iloc'):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        # Calculate weighted predictions from all estimators
        # Each estimator predicts its native labels (e.g., 0, 1)
        # Convert these to {-1, 1} before applying weights
        weighted_preds_sum = np.zeros(X_np.shape[0])
        for i, est in enumerate(self.estimators_):
            pred_orig_labels = est.predict(X_np)
            # Convert prediction to {-1, 1} using stored classes_
            pred_internal = np.where(pred_orig_labels == self.classes_[0], -1, 1)
            weighted_preds_sum += self.estimator_weights_[i] * pred_internal

        # Final prediction is the sign of the weighted sum
        y_pred_internal = np.sign(weighted_preds_sum).astype(int)
        # Handle cases where sum is exactly 0 (assign arbitrarily or based on prior)
        # A simple approach is to assign to one class, e.g., the second class (1)
        y_pred_internal[y_pred_internal == 0] = 1 # Or self.classes_[1] mapped to internal representation


        # Map back to original classes (e.g., {0, 1})
        y_pred = np.where(y_pred_internal == -1, self.classes_[0], self.classes_[1])

        return y_pred


# You might need to adjust the GradientBoostingRegressor class if it also uses
# the custom DecisionTreeRegressor and faces similar issues, although typically
# GBR fits trees to residuals (continuous values), which might not use bincount directly.
# Check your DecisionTreeRegressor implementation details if needed.

class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor Implementation.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of boosting stages to perform.
    learning_rate : float, optional (default=0.1)
        Learning rate shrinks the contribution of each tree.
    max_depth : int, optional (default=3)
        Maximum depth of the individual regression estimators.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node for the base learners.
    loss : str, optional (default='ls')
        The loss function to be optimized. 'ls' refers to least squares regression.
        (Currently only 'ls' is implemented).
    random_state : int, optional (default=None)
        Controls the random seed for reproducibility.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, loss='ls', random_state=None):
        if loss != 'ls':
            raise ValueError("Currently only 'ls' (Least Squares) loss is supported.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.random_state = random_state # Store the seed
        self.estimators_ = []
        self.initial_prediction_ = None
        # Initialize RNG here to be used across fit
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, X, y):
        """
        Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        # Convert to numpy arrays
        if hasattr(X, 'iloc'):
            X_np = X.values
        else:
            X_np = np.asarray(X)
        y_np = np.asarray(y)

        n_samples = X_np.shape[0]

        # Initial prediction: mean of target values
        self.initial_prediction_ = np.mean(y_np)
        current_predictions = np.full(n_samples, self.initial_prediction_)

        self.estimators_ = [] # Reset estimators list

        for i in range(self.n_estimators):
            # Compute residuals (negative gradient for squared error loss)
            residuals = y_np - current_predictions

            # Fit a regression tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                # Pass a new random state derived from the main RNG for each tree
                random_state=self._rng.integers(0, 1000000)
            )
            tree.fit(X_np, residuals) # Fit on residuals

            # Get predictions from the tree (leaf node values for residuals)
            tree_predictions = tree.predict(X_np)

            # Update current predictions
            current_predictions += self.learning_rate * tree_predictions

            # Store the fitted tree
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted values.
        """
        if self.initial_prediction_ is None or not self.estimators_:
             raise RuntimeError("GradientBoostingRegressor has not been fitted yet.")

        # Convert to numpy array
        if hasattr(X, 'iloc'):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        # Start with the initial prediction
        predictions = np.full(X_np.shape[0], self.initial_prediction_)

        # Add predictions from each tree, scaled by the learning rate
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X_np)

        return predictions