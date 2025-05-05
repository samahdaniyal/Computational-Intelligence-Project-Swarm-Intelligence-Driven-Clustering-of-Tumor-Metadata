import numpy as np
import random
from evaluation import evaluate_clustering
from sklearn.cluster import KMeans

class Ant:
    def __init__(self, X, pher, alpha, beta, eta):
        self.X, self.pher, self.alpha, self.beta, self.eta = X, pher, alpha, beta, eta
        self.n, self.k = X.shape[0], pher.shape[1]
        self.assignment, self.cost = None, None

    def construct(self):
        probs = (self.pher ** self.alpha) * (self.eta ** self.beta)
        probs /= probs.sum(1, keepdims=True)
        labels = np.array([random.choices(range(self.k), weights=probs[i])[0] for i in range(self.n)])

        # compute cluster centers and cost
        centers = np.array([self.X[labels == j].mean(0) if any(labels == j) else np.zeros(self.X.shape[1]) for j in range(self.k)])
        cost = np.sum((self.X - centers[labels]) ** 2)

        self.assignment, self.cost = labels, cost
        return labels, cost

class AntColonyClustering:
    def __init__(self, X, n_clusters, n_ants=20, iters=50, alpha_start=1.0, alpha_end=1.0, beta_start=2.0, beta_end=2.0, rho=0.1, Q=1.0):
        self.X, self.k = X, n_clusters
        self.n = X.shape[0]
        self.alpha_start, self.alpha_end = alpha_start, alpha_end
        self.beta_start, self.beta_end = beta_start, beta_end
        self.rho, self.Q = rho, Q
        self.pher = np.ones((self.n, self.k))
        self.n_ants, self.iters = n_ants, iters
        self.best, self.best_cost = None, np.inf

        # Better heuristic using KMeans++
        km = KMeans(n_clusters=self.k, init='k-means++', n_init=1).fit(self.X)
        centers = km.cluster_centers_
        distances = np.linalg.norm(self.X[:, None, :] - centers[None, :, :], axis=2)
        self.eta = 1.0 / (distances + 1e-10)

    def run(self):
        for it in range(self.iters):
            # Adaptive alpha and beta
            alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * it / (self.iters - 1)
            beta = self.beta_start + (self.beta_end - self.beta_start) * it / (self.iters - 1)

            all_sols = []
            for _ in range(self.n_ants):
                ant = Ant(self.X, self.pher, alpha, beta, self.eta)
                labels, cost = ant.construct()
                all_sols.append((labels, cost))

            # Find best ant
            best_labels, best_cost = min(all_sols, key=lambda x: x[1])
            if best_cost < self.best_cost:
                self.best, self.best_cost = best_labels.copy(), best_cost

            # Pheromone evaporation + normalization
            self.pher *= (1 - self.rho)
            self.pher = np.clip(self.pher, 1e-6, 10.0)

            # Elitist strategy: update only from best ant
            for i, lab in enumerate(best_labels):
                self.pher[i, lab] += self.Q / best_cost

            # Cluster-center repair: reassign empty clusters
            present_clusters = set(best_labels)
            empty = [j for j in range(self.k) if j not in present_clusters]
            for j in empty:
                i = np.argmax(self.pher[:, j])
                best_labels[i] = j

            print(f"[ACO] Iter {it+1}/{self.iters}, best_cost={self.best_cost:.4f}")

        return self.best, evaluate_clustering(self.X, self.best)

def run_aco(X: np.ndarray, n_clusters: int, **kwargs):
    print(f"[ACO] k={n_clusters}")
    algo = AntColonyClustering(X, n_clusters, **kwargs)
    return algo.run()