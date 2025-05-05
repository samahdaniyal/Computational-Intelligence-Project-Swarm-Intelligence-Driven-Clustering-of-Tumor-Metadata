import numpy as np
from evaluation import evaluate_clustering

class Particle:
    def __init__(self, dims, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dims)
        self.velocity = np.zeros(dims)
        self.best_pos = self.position.copy()
        self.best_cost = np.inf

    def evaluate(self, cost_func):
        cost = cost_func(self.position)
        if cost < self.best_cost:
            self.best_cost, self.best_pos = cost, self.position.copy()
        return cost

    def update(self, global_best, w, c1, c2, bounds):
        r1, r2 = np.random.rand(len(self.position)), np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_pos - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1])


def inertia_cost(flat_centroids, X, n_clusters):
    centers = flat_centroids.reshape((n_clusters, -1))
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)
    return np.sum((X - centers[labels])**2)

class PSOClustering:
    def __init__(self, X, n_clusters, n_particles=30, iters=100, w=0.7, c1=1.5, c2=1.5):
        self.X = X; self.k = n_clusters; self.iters = iters
        dims = self.k * X.shape[1]
        bounds = (np.tile(X.min(0), self.k), np.tile(X.max(0), self.k))
        self.particles = [Particle(dims, bounds) for _ in range(n_particles)]
        self.global_best, self.global_cost = None, np.inf
        self.w, self.c1, self.c2, self.bounds = w, c1, c2, bounds

    def optimize(self):
        for i in range(self.iters):
            for p in self.particles:
                cost = p.evaluate(lambda pos: inertia_cost(pos, self.X, self.k))
                if cost < self.global_cost:
                    self.global_cost, self.global_best = cost, p.position.copy()
            for p in self.particles:
                p.update(self.global_best, self.w, self.c1, self.c2, self.bounds)
            print(f"[PSO] Iter {i+1}/{self.iters}, best_cost={self.global_cost:.4f}")
        best_centers = self.global_best.reshape((self.k, -1))
        labels = np.argmin(np.linalg.norm(self.X[:, None, :] - best_centers[None,:,:], axis=2), axis=1)
        return labels, evaluate_clustering(self.X, labels)


def run_pso(X: np.ndarray, n_clusters: int, **kwargs):
    print(f"[PSO] k={n_clusters}")
    algo = PSOClustering(X, n_clusters, **kwargs)
    return algo.optimize()