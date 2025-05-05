import numpy as np

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    
    Parameters:
    -----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point (including the point itself).
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances.
        Currently only 'euclidean' is supported.
        
    Attributes:
    -----------
    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset.
        Noisy samples are given the label -1.
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.
    components_ : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.
    n_clusters_ : int
        The number of clusters found by the algorithm.
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.n_clusters_ = 0
        
    def fit(self, X):
        """
        Perform DBSCAN clustering on the data.
        
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
        
        # Check if the metric is supported
        if self.metric != 'euclidean':
            raise ValueError("Only 'euclidean' metric is currently supported.")
        
        # Initialize all points as unvisited
        visited = np.zeros(n_samples, dtype=bool)
        
        # Initialize all points as noise
        self.labels_ = np.full(n_samples, -1)
        
        # Initialize cluster counter
        cluster_id = 0
        
        # Visit each point
        for i in range(n_samples):
            # Skip visited points
            if visited[i]:
                continue
                
            # Mark as visited
            visited[i] = True
            
            # Find neighbors
            neighbors = self._region_query(X, i, X)
            
            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                # Expand the cluster
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1
        
        # Update the number of clusters
        self.n_clusters_ = cluster_id
        
        # Find core samples
        self.core_sample_indices_ = np.where(np.array([len(self._region_query(X, i, X)) >= self.min_samples for i in range(n_samples)]))[0]
        
        # Store core samples
        self.components_ = X[self.core_sample_indices_]
        
        return self
    
    def _region_query(self, X, point_idx, query_points):
        """
        Find all points within eps distance of a point.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Data set.
        point_idx : int
            Index of the point to find neighbors for.
        query_points : array-like, shape = [n_query_points, n_features]
            Points to calculate distance to.
            
        Returns:
        --------
        array
            Indices of neighboring points.
        """
        distances = np.linalg.norm(query_points - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        """
        Expand a cluster by adding all density-connected points.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Data set.
        point_idx : int
            Index of the point to expand from.
        neighbors : array
            Indices of neighboring points.
        cluster_id : int
            Cluster ID to assign.
        visited : array, shape = [n_samples]
            Boolean array indicating which points have been visited.
        """
        # Assign cluster ID to the seed point
        self.labels_[point_idx] = cluster_id
        
        # Process each neighbor
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If not visited, mark as visited
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                
                # Find neighbors of the neighbor
                neighbor_neighbors = self._region_query(X, neighbor_idx, X)
                
                # If enough neighbors, add them to the neighbor list
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.append(neighbors, neighbor_neighbors[~np.isin(neighbor_neighbors, neighbors)])
            
            # If not yet assigned to a cluster, add to this cluster
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
                
            i += 1
    
    def fit_predict(self, X):
        """
        Compute clusters and predict cluster index for each sample.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
            
        Returns:
        --------
        labels : array, shape = [n_samples]
            Cluster labels.
        """
        self.fit(X)
        return self.labels_ 