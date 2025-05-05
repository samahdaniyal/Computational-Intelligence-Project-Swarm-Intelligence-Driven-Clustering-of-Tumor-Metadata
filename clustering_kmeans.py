import numpy as np
from sklearn.cluster import KMeans
from evaluation import evaluate_clustering


def run_kmeans(X: np.ndarray, n_clusters: int, **kwargs) -> tuple[np.ndarray, dict]:
    print(f"[KMeans] Running with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, **kwargs)
    labels = km.fit_predict(X)
    print(f"[KMeans] Completed; found {len(np.unique(labels))} clusters.")
    return labels, evaluate_clustering(X, labels)
