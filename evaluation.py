import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray
) -> dict:
    unique = len(np.unique(labels))
    if unique < 2:
        print(f"[Evaluate] Only {unique} cluster(s); skipping metrics.")
        return {'silhouette': None, 'davies_bouldin': None}
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    print(f"[Evaluate] Silhouette={sil:.4f}, DB={db:.4f}")
    return {'silhouette': sil, 'davies_bouldin': db}