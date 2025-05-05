import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from clustering_kmeans import run_kmeans
from clustering_hierarchical import run_hierarchical
from clustering_pso import run_pso
from clustering_aco import run_aco
from cluster_selection import find_optimal_clusters
from visualize import plot_clustering_2d, plot_silhouette, plot_radar_chart, plot_cluster_distribution, plot_cluster_centers_heatmap, plot_convergence
import time

# load data:
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist. Please verify the path.")
    df = pd.read_csv(path, sep='\t')
    print(f"[DataLoader] Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# preprocess data:
def preprocess_data(df: pd.DataFrame, categorical_cols: list, numerical_cols: list) -> tuple[np.ndarray, OneHotEncoder, StandardScaler, pd.DataFrame]:
    missing = [c for c in categorical_cols + numerical_cols if c not in df.columns]
    if missing:
        print(f"[Preprocess] Warning: missing columns {missing}")
    df_sel = df[categorical_cols + numerical_cols].fillna('Missing')
    enc = OneHotEncoder(sparse_output=False)
    cat = enc.fit_transform(df_sel[categorical_cols])
    print(f"[Preprocess] Encoded categorical shape: {cat.shape}")
    scaler = StandardScaler()
    if numerical_cols:
        num = scaler.fit_transform(df_sel[numerical_cols])
        print(f"[Preprocess] Scaled numerical shape: {num.shape}")
    else:
        num = np.empty((len(df_sel), 0))
        print("[Preprocess] No numerical features.")
    X = np.hstack([cat, num])
    print(f"[Preprocess] Combined matrix X shape: {X.shape}")
    return X, enc, scaler, df_sel

# comparing clustering techniques:
def compare_methods(X, k, true_labels=None):
    results, perf, labels, conv = {}, {}, {}, {}
    for name, func in [('KMeans',run_kmeans),('Hierarchical',run_hierarchical),('PSO',run_pso),('ACO',run_aco)]:
        start=time.time()
        out=func(X,k)
        labels[name], conv[name] = out[0], out[2] if len(out)>2 else None
        perf[name]=time.time()-start
        results[name]=out[-1]
    return results, perf, labels, conv

# main:
# def main():
#     path = r"C:\Users\samah\Downloads\biospecimen.project-cmi-asc.2025-04-09\sample.tsv"
#     df = load_data(path)

#     cat_cols = [
#         'samples.biospecimen_anatomic_site',
#         'samples.composition',
#         'samples.tumor_descriptor'
#     ]
#     num_cols = []
#     X, _, _ = preprocess_data(df, cat_cols, num_cols)
#     k = 5

#     results = {}
#     for name, func in [
#         ('kmeans', run_kmeans),
#         ('hierarchical', run_hierarchical),
#         ('pso', run_pso),
#         ('aco', run_aco)
#     ]:
#         labels, ev = func(X, k)
#         results[name] = ev
#     print("Final results:", results)

# if __name__ == '__main__':
#     main()
    
# def main(path, cat_cols, num_cols, find_k=False):
#     df=load_data(path)
#     X,enc,scaler,_=preprocess_data(df,cat_cols,num_cols)
#     if find_k: print(find_optimal_clusters(X))
#     res,perf,labels,conv=compare_methods(X,5)
#     print(res)
#     plot_clustering_2d(X,labels)
#     for name in conv: plot_convergence(conv[name],name)
#     plot_cluster_centers_heatmap(X,labels)

# if __name__=='__main__':
#     main(r"CI Project\sample.tsv",['samples.biospecimen_anatomic_site', 'samples.composition','samples.tumor_descriptor'],[])
    
def main(path,cat_cols,num_cols,find_k=False):
    df=load_data(path)
    X,enc,scaler,_=preprocess_data(df,cat_cols,num_cols)
    if find_k: print(find_optimal_clusters(X))
    res,perf,labels,conv=compare_methods(X,5)
    print(res)
    plot_clustering_2d(X,labels,'pca')
    plot_clustering_2d(X,labels,'tsne')
    plot_silhouette(X,labels)
    plot_radar_chart(res)
    plot_cluster_distribution(labels)
    for name,h in conv.items():
        if h: plot_convergence(h,name)
    plot_cluster_centers_heatmap(X,labels)


if __name__ == '__main__':
    main(
        path=r"CI Project\sample.tsv",
        cat_cols=['samples.biospecimen_anatomic_site', 'samples.composition','samples.tumor_descriptor'],
        num_cols=[],
        find_k=False
    )