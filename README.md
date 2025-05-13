# Computational-Intelligence-Project-Swarm-Intelligence-Driven-Clustering-of-Tumor-Metadata

This repository contains the implementation of a comparative analysis of clustering algorithms applied to tumor metadata from the GDC Cancer Portal. The project evaluates four clustering techniques—K-means, Hierarchical Clustering, Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO)—to identify meaningful patterns in categorical tumor attributes. The goal is to uncover clinically relevant subgroups based on features like anatomic site, sample composition, and tumor descriptor.
The project includes data preprocessing, optimal cluster selection, clustering, evaluation using internal validation metrics, and comprehensive visualizations. It was developed as part of a final project at Habib University by Syeda Samah Daniyal, Aina Shakeel, and Zainab Raza.
Table of Contents

## Project Overview
Clustering tumor metadata is crucial for identifying patterns that can enhance cancer diagnosis and treatment strategies. This project compares traditional clustering methods (K-means, Hierarchical) with swarm intelligence-based approaches (PSO, ACO) to cluster 95 tumor samples based on categorical metadata. Key objectives include:

- Preprocessing categorical data using one-hot encoding.
- Determining the optimal number of clusters using the Elbow Method and Silhouette Analysis.
- Implementing and comparing four clustering algorithms.
- Evaluating clustering quality with internal metrics (Silhouette, Davies-Bouldin, Dunn, Calinski-Harabasz).
- Visualizing results with PCA/t-SNE projections, silhouette plots, radar charts, heatmaps, and cluster distributions.

The project highlights PSO's superior performance in cluster separation and biological interpretability, making it a promising approach for tumor metadata analysis.

## Dataset
The dataset (sample.tsv) contains 95 tumor samples from the GDC Cancer Portal, with the following categorical features:

- samples.biospecimen_anatomic_site: Anatomical location (e.g., Breast, Lung, Scalp).
- samples.composition: Sample type (e.g., Solid Tissue, Buffy Coat, Saliva).
- samples.tumor_descriptor: Tumor type (e.g., Primary, Metastatic, Recurrence, Not Applicable).

No numerical features were used due to the dataset's metadata-centric nature. The data is preprocessed by handling missing values (replaced with "Missing") and applying one-hot encoding to create a high-dimensional binary matrix.

## Features

Data Preprocessing: Handles missing values and converts categorical features to a numerical format using one-hot encoding.
Cluster Selection: Uses Elbow Method and Silhouette Analysis to determine the optimal number of clusters (_k_).
Clustering Algorithms:
- K-means: Fast, traditional method with k-means++ initialization.
- Hierarchical: Agglomerative clustering with Ward’s linkage.
- PSO: Custom implementation optimizing cluster centroids using particle swarm optimization.
- ACO: Custom implementation using pheromone-based probabilistic assignments.


Evaluation: Computes internal metrics (Silhouette, Davies-Bouldin, Dunn, Calinski-Harabasz) to assess clustering quality.
Visualization:
- PCA and t-SNE for 2D cluster projections.
- Silhouette plots for individual sample analysis.
- Radar charts for metric comparison.
- Cluster distribution histograms.
- Heatmaps of cluster centers for biological interpretability.

## Usage
To run the clustering analysis, execute the main script with the default parameters:
```
python main.py
```

This will:

- Load and preprocess the sample.tsv dataset.
- Determine the optimal (_k_) using find_optimal_clusters (Elbow and Silhouette methods).
- Run K-means, Hierarchical, PSO, and ACO clustering with the selected ( k ).
- Generate evaluation metrics and save visualizations as PNG files in the project directory.

Example Output

Console logs:
```
[DataLoader] Loaded data with 95 rows and X columns.
[Preprocess] Combined matrix X shape: (95, Y)
[ClusterSelection] Elbow Method suggests optimal k=4
[ClusterSelection] Silhouette Analysis suggests optimal k=5
[Main] Selected optimal k=5
```

Visualizations: PNG files like **clustering_2d_pca_KMeans.png, radar.png, heatmap_PSO.png**.

## Customizing Parameters
Edit the main.py script to modify:

- Dataset path: Change path=r"sample.tsv".
- Categorical columns: Adjust cat_cols list.
- Numerical columns: Add to num_cols if applicable.
- Cluster selection: Set find_k=False to use a fixed ( k ) (e.g., optimal_k=5).

## Results
The project evaluates clustering performance using internal validation metrics, as ground truth labels are unavailable. Key findings can be found in the project report, also available in this repository.

## Visualizations

- PCA/t-SNE Plots: PSO and K-means show distinct clusters, with PSO achieving 31% greater inter-cluster distance in PCA space.
- Heatmaps: PSO and K-means reveal biologically interpretable feature activations (e.g., separating Primary vs. Metastatic tumors).
- Radar Chart: Summarizes metric trade-offs, highlighting PSO’s balanced performance.
- Silhouette Plots: Hierarchical shows consistent cluster cohesion, while ACO exhibits variability.

See the **CI Final Project Report.pdf** for detailed analysis and visualization figures.

## Dependencies
The project requires the following Python packages:

- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0

Install them using:
```
pip install numpy pandas scikit-learn matplotlib seaborn
```

Authors: Syeda Samah Daniyal, Aina Shakeel, Zainab Raza
Email: sd07838@st.habib.edu.pk, as08430@st.habib.edu.pk, zr07532@st.habib.edu.pk
Institution: Department of Computer Science, Habib University, Karachi, Pakistan.
