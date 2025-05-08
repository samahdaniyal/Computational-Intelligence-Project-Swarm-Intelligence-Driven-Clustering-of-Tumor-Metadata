import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_clusters(X, k_range):
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertia.append(kmeans.inertia_)  # Within-cluster sum of squares
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.grid(True)
    plt.show()

    elbow_k = k_range[np.argmin(np.diff(inertia))] if len(inertia) > 1 else k_range[0]
    silhouette_k = k_range[np.argmax(silhouette_scores)]
    print(f"[ClusterSelection] Elbow Method suggests optimal k={elbow_k}")
    print(f"[ClusterSelection] Silhouette Analysis suggests optimal k={silhouette_k}")
    return max(elbow_k, silhouette_k)  # Use the higher of the two for a conservative estimate