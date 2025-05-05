import numpy as np
from sklearn.cluster import AgglomerativeClustering
from evaluation import evaluate_clustering


def run_hierarchical(X: np.ndarray, n_clusters: int, **kwargs) -> tuple[np.ndarray, dict]:
    print(f"[Hierarchical] Running with k={n_clusters}...")
    hc = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    labels = hc.fit_predict(X)
    print(f"[Hierarchical] Completed; found {len(np.unique(labels))} clusters.")
    return labels, evaluate_clustering(X, labels)