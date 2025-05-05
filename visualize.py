import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from collections import Counter

def plot_clustering_2d(X, labels_dict, method='pca'):
    reducer = PCA(n_components=2).fit_transform(X) if method == 'pca' else TSNE(n_components=2).fit_transform(X)
    for name, labels in labels_dict.items():
        plt.figure()
        plt.scatter(reducer[:, 0], reducer[:, 1], c=labels, alpha=0.6)
        plt.title(f'{method.upper()} - {name}')
        plt.savefig(f'clustering_2d_{method}_{name}.png')
        plt.close()

def plot_silhouette(X, labels_dict):
    from sklearn.metrics import silhouette_samples
    for name, labels in labels_dict.items():
        if len(np.unique(labels)) < 2:
            continue
        vals = silhouette_samples(X, labels)
        plt.figure(); y_lower = 0
        for cl in np.unique(labels):
            cl_vals = np.sort(vals[labels == cl])
            plt.barh(range(y_lower, y_lower + len(cl_vals)), cl_vals, height=1.0)
            y_lower += len(cl_vals)
        plt.title(f'Silhouette: {name}')
        plt.savefig(f'silhouette_{name}.png')
        plt.close()

def plot_radar_chart(results):
    metrics = ['silhouette', 'calinski_harabasz', 'dunn', 'davies_bouldin']
    methods = list(results.keys())
    data = []
    for m in metrics:
        vals = [results[met][m] if results[met][m] is not None else 0 for met in methods]
        if m == 'davies_bouldin':
            vals = [1 - v / max(vals) for v in vals]
        else:
            vals = [v / max(vals) if max(vals) > 0 else 0 for v in vals]
        data.append(vals)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for idx, met in enumerate(methods):
        vals = [d[idx] for d in data]
        vals += vals[:1]
        ax.plot(angles, vals, label=met)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.legend()
    plt.savefig('radar.png')
    plt.close()

def plot_cluster_distribution(labels_dict):
    plt.figure()
    for idx, (name, labels) in enumerate(labels_dict.items()):
        counts = Counter(labels)
        plt.bar(np.arange(len(counts)) + idx * 0.2, list(counts.values()), width=0.2, label=name)
    plt.legend()
    plt.savefig('cluster_dist.png')
    plt.close()

def plot_cluster_centers_heatmap(X, labels_dict):
    for name, labels in labels_dict.items():
        centers = np.array([X[labels == i].mean(0) for i in np.unique(labels)])
        plt.figure()
        sns.heatmap(centers, xticklabels=False)
        plt.title(f'Centers: {name}')
        plt.savefig(f'heatmap_{name}.png')
        plt.close()

def plot_convergence(conv, name):
    plt.figure()
    plt.plot(conv, label=name)
    plt.legend()
    plt.title(f'Convergence: {name}')
    plt.savefig(f'convergence_{name}.png')
    plt.close()
