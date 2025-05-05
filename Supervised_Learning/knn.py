import numpy as np
from collections import Counter

class KNNBase:
    """
    Base class for K-Nearest Neighbors algorithm.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction.
    weights : str, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform': all points in each neighborhood are weighted equally.
        - 'distance': weight points by the inverse of their distance.
    distance_metric : str, default='euclidean'
        Distance metric to use. Supported metrics: 'euclidean', 'manhattan'.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', distance_metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Fit the KNN model.
        
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
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
        
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points.
        
        Parameters:
        -----------
        x1, x2 : array-like
            Points to calculate distance between.
            
        Returns:
        --------
        float
            Distance between the points.
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Supported distance metrics are 'euclidean' and 'manhattan'")
            
    def _get_neighbors(self, x):
        """
        Find the k-nearest neighbors of a point.
        
        Parameters:
        -----------
        x : array-like
            Query point.
            
        Returns:
        --------
        tuple
            (indices, distances) of the k-nearest neighbors.
        """
        # Calculate distances
        distances = np.array([self._calculate_distance(x, x_train) for x_train in self.X_train])
        
        # Get indices of k-nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_distances = distances[k_indices]
        
        return k_indices, k_distances

class KNNClassifier(KNNBase):
    """
    K-Nearest Neighbors classifier.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for classification.
    weights : str, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform': all points in each neighborhood are weighted equally.
        - 'distance': weight points by the inverse of their distance.
    distance_metric : str, default='euclidean'
        Distance metric to use. Supported metrics: 'euclidean', 'manhattan'.
    """
    
    def predict(self, X):
        """
        Predict the class labels for the input samples.
        
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
        y_pred = np.array([self._predict_single(x) for x in X])
        return y_pred
        
    def _predict_single(self, x):
        """
        Predict the class for a single sample.
        
        Parameters:
        -----------
        x : array-like
            Input sample.
            
        Returns:
        --------
        object
            Predicted class label.
        """
        # Get k-nearest neighbors
        k_indices, k_distances = self._get_neighbors(x)
        k_targets = self.y_train[k_indices]
        
        # Weighted voting
        if self.weights == 'uniform':
            # Simple majority vote
            most_common = Counter(k_targets).most_common(1)
            return most_common[0][0]
        elif self.weights == 'distance':
            # Distance-weighted voting
            # Add small value to distances to avoid division by zero
            k_distances = k_distances + 1e-10
            weights = 1.0 / k_distances
            
            # Weighted votes for each class
            weighted_votes = {}
            for i, target in enumerate(k_targets):
                if target in weighted_votes:
                    weighted_votes[target] += weights[i]
                else:
                    weighted_votes[target] = weights[i]
            
            return max(weighted_votes.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Supported weight options are 'uniform' and 'distance'")
    
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
        # Find unique classes
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        
        # Initialize probabilities
        probabilities = np.zeros((X.shape[0], n_classes))
        
        # Get class probabilities for each sample
        for i, x in enumerate(X):
            # Get k-nearest neighbors
            k_indices, k_distances = self._get_neighbors(x)
            k_targets = self.y_train[k_indices]
            
            # Calculate weights
            if self.weights == 'uniform':
                weights = np.ones(self.n_neighbors)
            elif self.weights == 'distance':
                # Add small value to distances to avoid division by zero
                weights = 1.0 / (k_distances + 1e-10)
            else:
                raise ValueError("Supported weight options are 'uniform' and 'distance'")
            
            # Calculate class probabilities
            for j, cls in enumerate(classes):
                probabilities[i, j] = np.sum(weights[k_targets == cls]) / np.sum(weights)
                
        return probabilities

class KNNRegressor(KNNBase):
    """
    K-Nearest Neighbors regressor.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for regression.
    weights : str, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform': all points in each neighborhood are weighted equally.
        - 'distance': weight points by the inverse of their distance.
    distance_metric : str, default='euclidean'
        Distance metric to use. Supported metrics: 'euclidean', 'manhattan'.
    """
    
    def predict(self, X):
        """
        Predict target values for the input samples.
        
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
        y_pred = np.array([self._predict_single(x) for x in X])
        return y_pred
        
    def _predict_single(self, x):
        """
        Predict the target for a single sample.
        
        Parameters:
        -----------
        x : array-like
            Input sample.
            
        Returns:
        --------
        float
            Predicted target value.
        """
        # Get k-nearest neighbors
        k_indices, k_distances = self._get_neighbors(x)
        k_targets = self.y_train[k_indices]
        
        # Weighted average
        if self.weights == 'uniform':
            # Simple average
            return np.mean(k_targets)
        elif self.weights == 'distance':
            # Distance-weighted average
            # Add small value to distances to avoid division by zero
            k_distances = k_distances + 1e-10
            weights = 1.0 / k_distances
            return np.sum(weights * k_targets) / np.sum(weights)
        else:
            raise ValueError("Supported weight options are 'uniform' and 'distance'")
    
    def score(self, X, y):
        """
        Calculate R^2 (coefficient of determination) score.
        
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