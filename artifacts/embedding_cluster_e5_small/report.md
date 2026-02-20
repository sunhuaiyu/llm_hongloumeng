# Embedding Clustering Report

- Embedding model: `intfloat/multilingual-e5-small`
- Chunks embedded: `1171`
- Chapter embeddings: `120`
- Best method (by best_flip_accuracy): `gmm_2`

| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Label Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmm_2 | 0.0305 | 0.4379 | 0.3405 | 0.8333 | 76 | 18 | 27 |
| kmeans_2 | 0.0728 | -0.0016 | 0.0010 | 0.5500 | 16 | 42 | 40 |
| agglomerative_2 | 0.0576 | 0.0021 | 0.0128 | 0.5500 | 75 | 49 | 34 |

## Outputs
- Chunk embeddings: `artifacts/embedding_cluster_e5_small/chunk_embeddings.parquet`
- Chapter embeddings: `artifacts/embedding_cluster_e5_small/chapter_embeddings.parquet`
- Chapter clusters: `artifacts/embedding_cluster_e5_small/chapter_cluster_assignments.parquet`
- Metrics JSON: `artifacts/embedding_cluster_e5_small/cluster_metrics.json`
- Figure: `artifacts/embedding_cluster_e5_small/umap_best_cluster.png`
- Figure: `artifacts/embedding_cluster_e5_small/umap_true_period.png`
- Figure: `artifacts/embedding_cluster_e5_small/chapter_timeline_best_cluster.png`