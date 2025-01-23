#!/usr/bin/env python3
import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering on a dataset using scikit-learn.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Dataset of shape (n, d)
    k : int
        Number of clusters
    
    Returns:
    --------
    C : numpy.ndarray
        Centroid means of shape (k, d)
    clss : numpy.ndarray
        Cluster indices for each data point of shape (n,)
    """
    # Validate input
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    
    n, d = X.shape
    
    # Validate k
    if not isinstance(k, int) or k < 1 or k > n:
        return None, None
    
    try:
        # Create and fit K-means model
        kmeans = sklearn.cluster.KMeans(
            n_clusters=k, 
            n_init=10,  # Multiple initializations to avoid local optima
            random_state=42  # For reproducibility
        )
        kmeans.fit(X)
        
        # Extract cluster centroids
        C = kmeans.cluster_centers_
        
        # Get cluster assignments for each data point
        clss = kmeans.labels_
        
        return C, clss
    
    except Exception:
        return None, None
