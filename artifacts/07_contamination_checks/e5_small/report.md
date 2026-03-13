# Contamination-Resistant Checks

## Check 1: Corpus-Only TF-IDF Char-Ngram Unsupervised Clustering
| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmm_2 | 0.0271 | -0.0083 | 0.0099 | 0.6583 | 1 | 2 | 2 |
| kmeans_2 | 0.0000 | 0.0229 | 0.0132 | 0.5917 | 81 | 48 | 48 |
| agglomerative_2 | 0.0354 | -0.0211 | 0.0030 | 0.5750 | 15 | 20 | 27 |
- Null near-80 rate: `0.0000` (n=5000)

## Check 2: Name-Masked + Topic-Suppressed Embedding Unsupervised Clustering
- Embedding model: `intfloat/multilingual-e5-small`
- Topic terms removed: `400`
| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmm_2 | 0.0203 | 0.1254 | 0.1767 | 0.6833 | 74 | 36 | 39 |
| agglomerative_2 | 0.0231 | 0.0959 | 0.0515 | 0.6667 | 108 | 38 | 46 |
| kmeans_2 | 0.0281 | -0.0074 | 0.0088 | 0.5250 | 28 | 43 | 48 |
- Null near-80 rate: `0.0376` (n=5000)
- Observed best split: `74` (near 80: `False`)

## Check 3: Baseline E5 Label-Order Null Test
- Baseline best method: `gmm_2`
- Observed best split chapter: `76`
- Null near-80 rate: `0.0266`
- Null near-80 and error<=observed rate: `0.0000`

## Outputs
- Summary JSON: `artifacts/07_contamination_checks/e5_small/summary.json`
- Null histogram figure: `artifacts/07_contamination_checks/e5_small/null_split_hist_masked_embedding.png`