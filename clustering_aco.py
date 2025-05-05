import numpy as np
import random
from evaluation import evaluate_clustering

class Ant:
    def __init__(self, X, pher, alpha, beta):
        self.X, self.pher, self.alpha, self.beta = X, pher, alpha, beta
        self.n, self.k = X.shape[0], pher.shape[1]
        self.assignment, self.cost = None, None

    def construct(self):
        centers = np.random.randn(self.k, self.X.shape[1])  # placeholder
        # heuristic: inverse distance
        d = np.linalg.norm(self.X[:,None,:] - centers[None,:,:], axis=2)
        eta = 1/(d+1e-10)
        probs = (self.pher**self.alpha)*(eta**self.beta)
        probs /= probs.sum(1,keepdims=True)
        labels = np.array([random.choices(range(self.k), weights=probs[i])[0] for i in range(self.n)])
        # compute centers & cost
        new_centers = np.array([self.X[labels==j].mean(0) if any(labels==j) else np.zeros(self.X.shape[1]) for j in range(self.k)])
        cost = np.sum((self.X - new_centers[labels])**2)
        self.assignment, self.cost = labels, cost
        return labels, cost

class AntColonyClustering:
    def __init__(self, X, n_clusters, n_ants=20, iters=50, alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
        self.X, self.k = X, n_clusters
        self.n, self.alpha, self.beta, self.rho, self.Q = X.shape[0], alpha, beta, rho, Q
        self.pher = np.ones((self.n, self.k))
        self.n_ants, self.iters = n_ants, iters
        self.best, self.best_cost = None, np.inf

    def run(self):
        for it in range(self.iters):
            all_sols = []
            for _ in range(self.n_ants):
                ant = Ant(self.X, self.pher, self.alpha, self.beta)
                labels, cost = ant.construct()
                all_sols.append((labels, cost))
                if cost < self.best_cost:
                    self.best, self.best_cost = labels.copy(), cost
            # update pheromones
            self.pher *= (1-self.rho)
            for labels,cost in all_sols:
                for i,lab in enumerate(labels):
                    self.pher[i,lab] += self.Q/cost
            print(f"[ACO] Iter {it+1}/{self.iters}, best_cost={self.best_cost:.4f}")
        return self.best, evaluate_clustering(self.X, self.best)


def run_aco(X: np.ndarray, n_clusters: int, **kwargs):
    print(f"[ACO] k={n_clusters}")
    algo = AntColonyClustering(X, n_clusters, **kwargs)
    return algo.run()