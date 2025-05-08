import numpy as np
from sklearn.cluster import KMeans
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
        # clamp velocity
        v_max = 0.2 * (bounds[1] - bounds[0])
        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -v_max, v_max)
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1])


def inertia_cost(flat_centroids, X, n_clusters):
    centers = flat_centroids.reshape((n_clusters, -1))
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)
    return np.sum((X - centers[labels])**2)

class PSOClustering:
    def __init__(self, X, n_clusters, n_particles=30, iters=100, w=0.7, c1=1.5, c2=1.5, w_max=0.9, w_min=0.4):
        self.w_max, self.w_min = w_max, w_min
        self.X = X; self.k = n_clusters; self.iters = iters
        dims = self.k * X.shape[1]
        bounds = (np.tile(X.min(0), self.k), np.tile(X.max(0), self.k))
        self.particles = [Particle(dims, bounds) for _ in range(n_particles)]
        self.global_best, self.global_cost = None, np.inf
        self.w, self.c1, self.c2, self.bounds = w, c1, c2, bounds
        km = KMeans(n_clusters=self.k, init='k-means++', n_init=1).fit(self.X)
        seed = km.cluster_centers_.flatten()
        for i in range(min(n_particles, self.k)):
            noise = np.random.normal(0, 0.01, seed.shape)
            self.particles[i].position = np.clip(seed + noise, bounds[0], bounds[1])

    def optimize(self, patience= 20):
        history = []
        no_improve = 0
        for i in range(self.iters):
            # update inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * i / (self.iters - 1))
            history.append(self.global_cost)
            if i>0 and history[-1] >= history[-2]:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= patience:
                print(f"[PSO] Early stopping at iter {i+1}")
                break
            for p in self.particles:
                cost = p.evaluate(lambda pos: inertia_cost(pos, self.X, self.k))
                if cost < self.global_cost:
                    self.global_cost, self.global_best = cost, p.position.copy()
            for p in self.particles:
                p.update(self.global_best, self.w, self.c1, self.c2, self.bounds)
            print(f"[PSO] Iter {i+1}/{self.iters}, best_cost={self.global_cost:.4f}")
        best_centers = self.global_best.reshape((self.k, -1))
        km = KMeans(n_clusters=self.k, init=best_centers, n_init=1).fit(self.X)
        labels = km.labels_
        return labels, evaluate_clustering(self.X, labels)


def run_pso(X: np.ndarray, n_clusters: int, **kwargs):
    print(f"[PSO] k={n_clusters}")
    algo = PSOClustering(X, n_clusters, **kwargs)
    return algo.optimize()