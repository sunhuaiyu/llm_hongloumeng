# Contamination-Resistant Checks

## Check 1: Corpus-Only TF-IDF Char-Ngram Unsupervised Clustering
| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmm_2 | 0.0271 | -0.0083 | 0.0099 | 0.6583 | 1 | 2 | 2 |
| kmeans_2 | 0.0000 | 0.0229 | 0.0132 | 0.5917 | 81 | 48 | 48 |
| agglomerative_2 | 0.0354 | -0.0211 | 0.0030 | 0.5750 | 15 | 20 | 27 |
- Null near-80 rate: `0.0000` (n=5000)

## Check 2: Name-Masked + Topic-Suppressed Embedding Unsupervised Clustering
- Embedding model: `BAAI/bge-small-zh-v1.5`
- Topic terms removed: `400`
| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| agglomerative_2 | 0.0310 | 0.1101 | 0.2083 | 0.6750 | 19 | 34 | 35 |
| kmeans_2 | 0.0282 | 0.0397 | 0.0425 | 0.6083 | 19 | 46 | 53 |
| gmm_2 | 0.0321 | 0.0137 | 0.1031 | 0.5917 | 1 | 40 | 42 |
- Null near-80 rate: `0.0136` (n=5000)
- Observed best split: `19` (near 80: `False`)

## Check 3: Baseline E5 Label-Order Null Test
- Baseline best method: `gmm_2`
- Observed best split chapter: `76`
- Null near-80 rate: `0.0266`
- Null near-80 and error<=observed rate: `0.0000`

## Outputs
- Summary JSON: `artifacts/contamination_checks_bge_small_zh/summary.json`
- Null histogram figure: `artifacts/contamination_checks_bge_small_zh/null_split_hist_masked_embedding.png`