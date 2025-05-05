import numpy as np

class SVDCompression:
    """
    Singular Value Decomposition (SVD) for image compression.
    
    Parameters:
    -----------
    n_components : int, default=None
        Number of singular values to keep.
        If None, all components are kept.
    
    Attributes:
    -----------
    U_ : array
        Left singular vectors.
    sigma_ : array
        Singular values.
    Vt_ : array
        Right singular vectors (transposed).
    explained_variance_ratio_ : array
        Percentage of variance explained by each component.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U_ = None
        self.sigma_ = None
        self.Vt_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """
        Compute the SVD decomposition of the input matrix.
        
        Parameters:
        -----------
        X : array-like, shape = [height, width] or [height, width, channels]
            Input image.
            
        Returns:
        --------
        self : object
        """
        # Convert input to numpy array
        X = np.array(X)
        
        # Determine if the image is grayscale or color
        self.is_color = len(X.shape) == 3 and X.shape[2] > 1
        
        # If color image, handle each channel separately
        if self.is_color:
            self.channels = X.shape[2]
            self.U_ = []
            self.sigma_ = []
            self.Vt_ = []
            self.explained_variance_ratio_ = []
            
            for i in range(self.channels):
                channel = X[:, :, i]
                U, sigma, Vt, explained_variance_ratio = self._svd_channel(channel)
                self.U_.append(U)
                self.sigma_.append(sigma)
                self.Vt_.append(Vt)
                self.explained_variance_ratio_.append(explained_variance_ratio)
        else:
            # For grayscale images
            self.U_, self.sigma_, self.Vt_, self.explained_variance_ratio_ = self._svd_channel(X)
            
        return self
    
    def _svd_channel(self, channel):
        """
        Compute SVD for a single channel.
        
        Parameters:
        -----------
        channel : array, shape = [height, width]
            Single channel of the image.
            
        Returns:
        --------
        tuple
            (U, sigma, Vt, explained_variance_ratio)
        """
        # Compute full SVD
        U, sigma, Vt = np.linalg.svd(channel, full_matrices=False)
        
        # Calculate variance explained by each component
        explained_variance = (sigma ** 2) / (channel.shape[0] - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
        
        # Determine number of components to keep
        if self.n_components is not None:
            n_components = min(self.n_components, len(sigma))
            U = U[:, :n_components]
            sigma = sigma[:n_components]
            Vt = Vt[:n_components, :]
            explained_variance_ratio = explained_variance_ratio[:n_components]
        
        return U, sigma, Vt, explained_variance_ratio
    
    def transform(self, n_components=None):
        """
        Transform the input using the SVD decomposition with the specified number of components.
        
        Parameters:
        -----------
        n_components : int, default=None
            Number of components to use in the transformation.
            If None, the number of components specified in the constructor is used.
            
        Returns:
        --------
        X_compressed : array
            Compressed image.
        """
        if self.U_ is None:
            raise ValueError("Model must be fitted before calling transform")
            
        # Use the constructor's n_components if none specified
        if n_components is None:
            n_components = self.n_components
            
        # If no specific n_components was set, use all
        if n_components is None:
            # Return the original image reconstruction
            if self.is_color:
                return self._reconstruct_color_image()
            else:
                return self._reconstruct_grayscale_image()
        
        # Otherwise, reconstruct with specified number of components
        if self.is_color:
            return self._reconstruct_color_image(n_components)
        else:
            return self._reconstruct_grayscale_image(n_components)
    
    def _reconstruct_grayscale_image(self, n_components=None):
        """
        Reconstruct a grayscale image using the specified number of components.
        
        Parameters:
        -----------
        n_components : int, default=None
            Number of components to use in the reconstruction.
            If None, all components are used.
            
        Returns:
        --------
        reconstructed : array, shape = [height, width]
            Reconstructed grayscale image.
        """
        U, sigma, Vt = self.U_, self.sigma_, self.Vt_
        
        if n_components is not None:
            n_components = min(n_components, len(sigma))
            U = U[:, :n_components]
            sigma = sigma[:n_components]
            Vt = Vt[:n_components, :]
        
        # Reconstruct the image
        reconstructed = np.dot(U * sigma, Vt)
        
        return reconstructed
    
    def _reconstruct_color_image(self, n_components=None):
        """
        Reconstruct a color image using the specified number of components.
        
        Parameters:
        -----------
        n_components : int, default=None
            Number of components to use in the reconstruction.
            If None, all components are used.
            
        Returns:
        --------
        reconstructed : array, shape = [height, width, channels]
            Reconstructed color image.
        """
        # Reconstruct each channel separately
        channels = []
        
        for i in range(self.channels):
            U, sigma, Vt = self.U_[i], self.sigma_[i], self.Vt_[i]
            
            if n_components is not None:
                n_components_i = min(n_components, len(sigma))
                U = U[:, :n_components_i]
                sigma = sigma[:n_components_i]
                Vt = Vt[:n_components_i, :]
            
            # Reconstruct the channel
            channel = np.dot(U * sigma, Vt)
            channels.append(channel)
        
        # Combine channels
        reconstructed = np.stack(channels, axis=2)
        
        return reconstructed
    
    def compression_ratio(self, n_components=None):
        """
        Calculate the compression ratio achieved.
        
        Parameters:
        -----------
        n_components : int, default=None
            Number of components used in the compression.
            If None, the number of components specified in the constructor is used.
            
        Returns:
        --------
        ratio : float
            Compression ratio (original size / compressed size).
        """
        if self.U_ is None:
            raise ValueError("Model must be fitted before calculating compression ratio")
            
        # Use the constructor's n_components if none specified
        if n_components is None:
            n_components = self.n_components
            
        # If still None (no specific n_components was set), there's no compression
        if n_components is None:
            return 1.0
            
        if self.is_color:
            # Calculate for the first channel as an example
            U, sigma, Vt = self.U_[0], self.sigma_[0], self.Vt_[0]
            height, width = U.shape[0], Vt.shape[1]
            original_size = height * width * self.channels
            
            # Size with compression
            n_comp = min(n_components, len(sigma))
            compressed_size = (height * n_comp + n_comp + n_comp * width) * self.channels
        else:
            # For grayscale
            U, sigma, Vt = self.U_, self.sigma_, self.Vt_
            height, width = U.shape[0], Vt.shape[1]
            original_size = height * width
            
            # Size with compression
            n_comp = min(n_components, len(sigma))
            compressed_size = height * n_comp + n_comp + n_comp * width
            
        return original_size / compressed_size
    
    def get_explained_variance(self, n_components=None):
        """
        Get the cumulative explained variance for the specified number of components.
        
        Parameters:
        -----------
        n_components : int, default=None
            Number of components to consider.
            If None, all components are considered.
            
        Returns:
        --------
        explained_variance : float or list
            Cumulative explained variance. If color image, returns a list for each channel.
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model must be fitted before getting explained variance")
            
        if n_components is None:
            n_components = len(self.explained_variance_ratio_[0]) if self.is_color else len(self.explained_variance_ratio_)
            
        if self.is_color:
            return [np.sum(ratio[:min(n_components, len(ratio))]) for ratio in self.explained_variance_ratio_]
        else:
            return np.sum(self.explained_variance_ratio_[:min(n_components, len(self.explained_variance_ratio_))]) 