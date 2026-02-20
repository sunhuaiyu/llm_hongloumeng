# Embedding Clustering Report

- Embedding model: `shibing624/text2vec-base-chinese-paraphrase`
- Chunks embedded: `1171`
- Chapter embeddings: `120`
- Best method (by best_flip_accuracy): `agglomerative_2`

| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Label Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| agglomerative_2 | 0.0760 | -0.0016 | 0.0256 | 0.5500 | 38 | 42 | 35 |
| gmm_2 | 0.0937 | -0.0064 | 0.0009 | 0.5167 | 38 | 50 | 41 |
| kmeans_2 | 0.0930 | -0.0076 | 0.0001 | 0.5000 | 38 | 48 | 37 |

## Outputs
- Chunk embeddings: `artifacts/embedding_cluster_text2vec_zh/chunk_embeddings.parquet`
- Chapter embeddings: `artifacts/embedding_cluster_text2vec_zh/chapter_embeddings.parquet`
- Chapter clusters: `artifacts/embedding_cluster_text2vec_zh/chapter_cluster_assignments.parquet`
- Metrics JSON: `artifacts/embedding_cluster_text2vec_zh/cluster_metrics.json`
- Figure: `artifacts/embedding_cluster_text2vec_zh/umap_best_cluster.png`
- Figure: `artifacts/embedding_cluster_text2vec_zh/umap_true_period.png`
- Figure: `artifacts/embedding_cluster_text2vec_zh/chapter_timeline_best_cluster.png`