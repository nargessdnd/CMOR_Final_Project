import numpy as np

class KMeans:
    """
    K-Means clustering algorithm.
    
    Parameters:
    -----------
    n_clusters : int, default=8
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations for a single run.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Attributes:
    -----------
    centroids_ : array, shape = [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ : array, shape = [n_samples]
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
    """
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def fit(self, X):
        """
        Compute k-means clustering.
        
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
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Initialize centroids by randomly selecting data points
        idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids_ = X[idx].copy()
        
        # Initialize previous centroids
        prev_centroids = np.zeros_like(self.centroids_)
        
        # Initialize labels
        self.labels_ = np.zeros(n_samples)
        
        # Main loop
        for i in range(self.max_iter):
            self.n_iter_ += 1
            
            # Assign labels based on closest centroid
            for j in range(n_samples):
                distances = np.linalg.norm(X[j] - self.centroids_, axis=1)
                self.labels_[j] = np.argmin(distances)
            
            # Store previous centroids
            prev_centroids = self.centroids_.copy()
            
            # Update centroids based on mean of points in each cluster
            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:  # Check if cluster is not empty
                    self.centroids_[k] = np.mean(X[self.labels_ == k], axis=0)
            
            # Check for convergence
            if np.linalg.norm(self.centroids_ - prev_centroids) < self.tol:
                break
        
        # Calculate inertia (sum of squared distances to closest centroid)
        self.inertia_ = 0.0
        for j in range(n_samples):
            self.inertia_ += np.linalg.norm(X[j] - self.centroids_[int(self.labels_[j])]) ** 2
            
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
            
        Returns:
        --------
        labels : array, shape = [n_samples]
            Index of the cluster each sample belongs to.
        """
        # Convert input to numpy array
        X = np.array(X)
        
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)
        
        # Assign labels based on closest centroid
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - self.centroids_, axis=1)
            labels[i] = np.argmin(distances)
            
        return labels
    
    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
            
        Returns:
        --------
        labels : array, shape = [n_samples]
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_
    
    def transform(self, X):
        """
        Transform X to a cluster-distance space.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            New data to transform.
            
        Returns:
        --------
        X_new : array, shape = [n_samples, n_clusters]
            X transformed in the new space.
        """
        # Convert input to numpy array
        X = np.array(X)
        
        n_samples = X.shape[0]
        result = np.zeros((n_samples, self.n_clusters))
        
        # Calculate distance to each centroid
        for i in range(n_samples):
            for j in range(self.n_clusters):
                result[i, j] = np.linalg.norm(X[i] - self.centroids_[j])
                
        return result 