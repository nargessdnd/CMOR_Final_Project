# Supervised_Learning/ensemble_methods.py

import numpy as np
from sklearn.base import clone  # Use clone for sklearn compatibility if needed later
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier # For default if none provided
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
        """
        params = {}
        if hasattr(estimator, 'max_depth'):
            params['max_depth'] = estimator.max_depth
        if hasattr(estimator, 'min_samples_split'):
            params['min_samples_split'] = estimator.min_samples_split
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
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values. Assumes binary {-1, 1} or {0, 1}. Will convert {0, 1} to {-1, 1}.
        """
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Ensure y is in {-1, 1} format internally for AdaBoost calculations
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [-1, 1]):
            y_internal = np.where(y == self.classes_[0], -1, 1) # Assumes binary
        else:
            y_internal = y.copy()


        # Initialize weights
        sample_weights = np.full(n_samples, (1 / n_samples))

        for iboost in range(self.n_estimators):
            # --- FIX START ---
            # Create a new instance of the base estimator for this iteration
            # Manually pass known parameters instead of relying on get_params()
            estimator_params = self._get_estimator_params(self.base_estimator_)
            # Add random_state for this specific estimator if base supports it
            if 'random_state' in estimator_params or isinstance(self.base_estimator_, DecisionTreeClassifier): # Check if our DT supports random_state
                 estimator_params['random_state'] = rng.integers(0, 1000000)

            # Instantiate the estimator using its type and the collected parameters
            estimator = type(self.base_estimator_)(**estimator_params)
            # --- FIX END ---


            # Fit the estimator using weighted bootstrap sampling
            # (since our DT doesn't directly support sample_weight)
            bootstrap_idxs = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights / np.sum(sample_weights) # Ensure weights sum to 1
            )
            X_bootstrap, y_bootstrap = X.iloc[bootstrap_idxs], y_internal[bootstrap_idxs]

            # Fit the weak learner
            estimator.fit(X_bootstrap, y_bootstrap)
            y_pred = estimator.predict(X)

            # Calculate estimator error
            incorrect = (y_pred != y_internal)
            estimator_error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            # Avoid division by zero and handle perfect classifiers
            if estimator_error <= 0:
                # Stop if perfect fit early
                self.estimator_weights_[iboost] = 1.0
                self.estimator_errors_[iboost] = 0.0
                self.estimators_.append(estimator)
                break
            elif estimator_error >= 0.5:
                 # Skip if worse than random guessing (or handle differently, e.g., break)
                 # For simplicity, we might just get low weight and continue
                 # AdaBoost theory suggests breaking or handling differently if error > 0.5
                 # Here we let alpha calculation handle it (alpha becomes negative or zero)
                 pass


            # Calculate estimator weight (alpha)
            # Avoid log(0) or log(negative)
            eps = 1e-10 # Small epsilon for numerical stability
            alpha = self.learning_rate * 0.5 * np.log((1.0 - estimator_error + eps) / (estimator_error + eps))


            # Update sample weights
            sample_weights *= np.exp(-alpha * y_internal * y_pred)
            # Normalize sample weights
            sample_weights /= np.sum(sample_weights)

            # Store the estimator and its weight
            self.estimators_.append(estimator)
            self.estimator_weights_[iboost] = alpha
            self.estimator_errors_[iboost] = estimator_error

            # Early stopping if only one estimator or error is too high
            if estimator_error >= 0.5 and len(self.estimators_) > 1:
                 print(f"Estimator error >= 0.5 ({estimator_error:.4f}) at iteration {iboost}. Stopping early.")
                 break


        # Trim unused estimator slots if we stopped early
        actual_n_estimators = len(self.estimators_)
        self.estimator_weights_ = self.estimator_weights_[:actual_n_estimators]
        self.estimator_errors_ = self.estimator_errors_[:actual_n_estimators]

        return self

    def predict(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted classes.
        """
        # Calculate weighted predictions from all estimators
        # Note: Assumes predict returns {-1, 1}
        estimator_preds = np.array([est.predict(X) for est in self.estimators_])
        weighted_preds = np.dot(self.estimator_weights_, estimator_preds)

        # Final prediction is the sign of the weighted sum
        y_pred_internal = np.sign(weighted_preds).astype(int)

        # Map back to original classes {0, 1} if necessary
        if not np.array_equal(self.classes_, [-1, 1]):
             y_pred = np.where(y_pred_internal == -1, self.classes_[0], self.classes_[1])
        else:
             y_pred = y_pred_internal

        return y_pred


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
        self.random_state = random_state
        self.estimators_ = []
        self.initial_prediction_ = None
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
        n_samples = X.shape[0]

        # Initial prediction: mean of target values
        self.initial_prediction_ = np.mean(y)
        current_predictions = np.full(n_samples, self.initial_prediction_)

        for i in range(self.n_estimators):
            # Compute residuals (negative gradient for squared error loss)
            residuals = y - current_predictions

            # Fit a regression tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self._rng.integers(0, 1000000) # Pass random state to tree
            )
            tree.fit(X, residuals)

            # Get predictions from the tree (leaf node values for residuals)
            tree_predictions = tree.predict(X)

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
        # Start with the initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction_)

        # Add predictions from each tree, scaled by the learning rate
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)

        return predictions