import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from clustering_kmeans import run_kmeans
from clustering_hierarchical import run_hierarchical
from clustering_pso import run_pso
from clustering_aco import run_aco

# load data:
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist. Please verify the path.")
    df = pd.read_csv(path, sep='\t')
    print(f"[DataLoader] Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# preprocess data:
def preprocess_data(df: pd.DataFrame, categorical_cols: list, numerical_cols: list) -> tuple[np.ndarray, OneHotEncoder, StandardScaler]:
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
    return X, enc, scaler

# main:
def main():
    path = r"C:\Users\samah\Downloads\biospecimen.project-cmi-asc.2025-04-09\sample.tsv"
    df = load_data(path)

    cat_cols = [
        'samples.biospecimen_anatomic_site',
        'samples.composition',
        'samples.tumor_descriptor'
    ]
    num_cols = []
    X, _, _ = preprocess_data(df, cat_cols, num_cols)
    k = 5

    results = {}
    for name, func in [
        ('kmeans', run_kmeans),
        ('hierarchical', run_hierarchical),
        ('pso', run_pso),
        ('aco', run_aco)
    ]:
        labels, ev = func(X, k)
        results[name] = ev
    print("Final results:", results)

if __name__ == '__main__':
    main()