# Unsupervised Learning Algorithms

This directory contains implementations of various unsupervised learning algorithms, all custom-built using NumPy and basic Python libraries.

## Algorithms Implemented

- **K-Means Clustering**: Partitioning method that divides the dataset into k distinct clusters
- **DBSCAN**: Density-based spatial clustering algorithm for discovering clusters of arbitrary shape
- **Principal Component Analysis (PCA)**: Dimension reduction technique that identifies the axes of maximum variance
- **Singular Value Decomposition (SVD)**: Matrix factorization technique used for dimensionality reduction and image compression
- **Label Propagation**: Semi-supervised learning algorithm that propagates labels through a similarity graph

## Dataset Usage

These algorithms are applied to the ESG (Environmental, Social, and Governance) and Financial Performance dataset within the main `ESG_Financial_Analysis.ipynb` notebook. The SVD implementation is demonstrated with image compression examples using the CIFAR-10 dataset.

## Usage Example

```python
from Unsupervised_Learning.kmeans import KMeans

# Create and apply clustering
kmeans = KMeans(n_clusters=3, max_iters=100)
cluster_assignments = kmeans.fit_predict(X)
```

## Applications in ESG Analysis

The unsupervised learning techniques are used to:
1. Group companies with similar ESG profiles (clustering)
2. Reduce dimensionality of complex ESG metrics (PCA)
3. Identify underlying patterns in financial and ESG data
4. Propagate ESG ratings from labeled to unlabeled companies (label propagation)

## Reproducing Results

Refer to the `ESG_Financial_Analysis.ipynb` notebook for details on data preprocessing, model application, and results analysis using these implementations. 