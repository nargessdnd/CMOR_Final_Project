import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA).
    
    PCA performs dimensionality reduction by finding the orthogonal directions
    of maximum variance in the data.
    
    Parameters:
    -----------
    n_components : int or float, default=None
        Number of components to keep.
        - If n_components is an integer, it represents the number of components to keep.
        - If n_components is a float between 0 and 1, it represents the minimum fraction
          of variance to be retained.
        - If n_components is None, all components are kept.
    whiten : bool, default=False
        When True, the components_ vectors are divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
    random_state : int, default=42
        Random seed for reproducibility, used for randomized algorithms.
        
    Attributes:
    -----------
    components_ : array, shape = [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    explained_variance_ : array, shape = [n_components]
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : array, shape = [n_components]
        Percentage of variance explained by each of the selected components.
    mean_ : array, shape = [n_features]
        Per-feature empirical mean, estimated from the training set.
    n_components_ : int
        The actual number of components used in the transformation.
    """
    
    def __init__(self, n_components=None, whiten=False, random_state=42):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_components_ = None
        
    def fit(self, X):
        """
        Fit the model with X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
            
        Returns:
        --------
        self : object
        """
        # Convert input to numpy array
        X = np.array(X)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute the covariance matrix
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine the number of components to keep
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            # Calculate the number of components needed to explain the desired variance
            total_variance = np.sum(eigenvalues)
            cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance
            self.n_components_ = np.searchsorted(cumulative_variance_ratio, self.n_components) + 1
        else:
            self.n_components_ = min(self.n_components, min(n_samples, n_features))
        
        # Store components, explained variance, and explained variance ratio
        self.components_ = eigenvectors[:, :self.n_components_].T
        self.explained_variance_ = eigenvalues[:self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        # Apply whitening if requested
        if self.whiten:
            self.components_ = self.components_ / np.sqrt(self.explained_variance_[:, np.newaxis])
            
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            New data.
            
        Returns:
        --------
        X_new : array, shape = [n_samples, n_components]
            Transformed values.
        """
        # Convert input to numpy array
        X = np.array(X)
        
        # Center the data using the training mean
        X_centered = X - self.mean_
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply dimensionality reduction on X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
            
        Returns:
        --------
        X_new : array, shape = [n_samples, n_components]
            Transformed values.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_components]
            Transformed data.
            
        Returns:
        --------
        X_original : array, shape = [n_samples, n_features]
            Data in original space.
        """
        # Convert input to numpy array
        X = np.array(X)
        
        # Project back to the original space
        X_original = np.dot(X, self.components_) + self.mean_
        
        return X_original 