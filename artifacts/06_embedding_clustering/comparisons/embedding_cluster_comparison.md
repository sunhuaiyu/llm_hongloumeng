# Embedding Clustering Comparison

| Run | Model | Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| bge_small_zh | BAAI/bge-small-zh-v1.5 | gmm_2 | 0.0866 | 0.0048 | 0.0052 | 0.5583 | 18 | 41 | 47 |
| bge_small_zh | BAAI/bge-small-zh-v1.5 | agglomerative_2 | 0.0757 | -0.0053 | 0.0009 | 0.5250 | 38 | 51 | 43 |
| bge_small_zh | BAAI/bge-small-zh-v1.5 | kmeans_2 | 0.0835 | -0.0065 | 0.0004 | 0.5167 | 16 | 48 | 43 |
| e5_small | intfloat/multilingual-e5-small | gmm_2 | 0.0305 | 0.4379 | 0.3405 | 0.8333 | 76 | 18 | 27 |
| e5_small | intfloat/multilingual-e5-small | kmeans_2 | 0.0728 | -0.0016 | 0.0010 | 0.5500 | 16 | 42 | 40 |
| e5_small | intfloat/multilingual-e5-small | agglomerative_2 | 0.0576 | 0.0021 | 0.0128 | 0.5500 | 75 | 49 | 34 |
| text2vec_zh | shibing624/text2vec-base-chinese-paraphrase | agglomerative_2 | 0.0760 | -0.0016 | 0.0256 | 0.5500 | 38 | 42 | 35 |
| text2vec_zh | shibing624/text2vec-base-chinese-paraphrase | gmm_2 | 0.0937 | -0.0064 | 0.0009 | 0.5167 | 38 | 50 | 41 |
| text2vec_zh | shibing624/text2vec-base-chinese-paraphrase | kmeans_2 | 0.0930 | -0.0076 | 0.0001 | 0.5000 | 38 | 48 | 37 |