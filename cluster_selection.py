from sklearn.cluster import KMeans, AgglomerativeClustering
from evaluation import evaluate_clustering
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_clusters(X, max_k=10, method='kmeans'):
    ks, sil, db, ch = [], [], [], []
    for k in range(2, max_k+1):
        model = KMeans(n_clusters=k) if method=='kmeans' else AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
        ks.append(k)
        sil.append(evaluate_clustering(X,labels)['silhouette'])
        db.append(evaluate_clustering(X,labels)['davies_bouldin'])
        ch.append(evaluate_clustering(X,labels)['calinski_harabasz'])
    plt.plot(ks,sil,label='Silhouette')
    plt.plot(ks,db,label='DB')
    plt.plot(ks,ch,label='Calinski-Harabasz')
    plt.legend(); plt.show()
    optimal={'silhouette':ks[np.nanargmax(sil)],'davies_bouldin':ks[np.nanargmin(db)],'calinski_harabasz':ks[np.nanargmax(ch)]}
    return optimal