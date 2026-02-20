# Embedding Clustering Report

- Embedding model: `BAAI/bge-small-zh-v1.5`
- Chunks embedded: `1171`
- Chapter embeddings: `120`
- Best method (by best_flip_accuracy): `gmm_2`

| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Label Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmm_2 | 0.0866 | 0.0048 | 0.0052 | 0.5583 | 18 | 41 | 47 |
| agglomerative_2 | 0.0757 | -0.0053 | 0.0009 | 0.5250 | 38 | 51 | 43 |
| kmeans_2 | 0.0835 | -0.0065 | 0.0004 | 0.5167 | 16 | 48 | 43 |

## Outputs
- Chunk embeddings: `artifacts/embedding_cluster_bge_small_zh/chunk_embeddings.parquet`
- Chapter embeddings: `artifacts/embedding_cluster_bge_small_zh/chapter_embeddings.parquet`
- Chapter clusters: `artifacts/embedding_cluster_bge_small_zh/chapter_cluster_assignments.parquet`
- Metrics JSON: `artifacts/embedding_cluster_bge_small_zh/cluster_metrics.json`
- Figure: `artifacts/embedding_cluster_bge_small_zh/umap_best_cluster.png`
- Figure: `artifacts/embedding_cluster_bge_small_zh/umap_true_period.png`
- Figure: `artifacts/embedding_cluster_bge_small_zh/chapter_timeline_best_cluster.png`