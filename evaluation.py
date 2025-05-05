import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    k = len(unique)
    if k < 2:
        return None
    centroids = np.array([X[labels==i].mean(0) for i in unique])
    inter = np.min([np.linalg.norm(centroids[i]-centroids[j])
                    for i in range(k) for j in range(i+1, k)])
    intra = np.max([np.linalg.norm(p-q)
                    for i in unique
                    for p in X[labels==i]
                    for q in X[labels==i] if not np.array_equal(p, q)])
    return inter/intra if intra>0 else None


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, true_labels: np.ndarray=None) -> dict:
    results = {}
    k = len(np.unique(labels))
    # Internal metrics
    if k > 1:
        results['silhouette'] = silhouette_score(X, labels)
        results['davies_bouldin'] = davies_bouldin_score(X, labels)
        results['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        results['dunn'] = dunn_index(X, labels)
    else:
        for m in ['silhouette','davies_bouldin','calinski_harabasz','dunn']:
            results[m] = None
    # External metrics
    if true_labels is not None:
        from sklearn.metrics import (
            adjusted_rand_score,normalized_mutual_info_score, f1_score
        )
        results['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
        results['nmi'] = normalized_mutual_info_score(true_labels, labels)
        if len(np.unique(true_labels))==len(np.unique(labels)):
            try:
                results['f1'] = f1_score(true_labels, labels, average='weighted')
            except:
                results['f1'] = None
        else:
            results['f1'] = None
    return results