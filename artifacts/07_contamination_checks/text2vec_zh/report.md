# Contamination-Resistant Checks

## Check 1: Corpus-Only TF-IDF Char-Ngram Unsupervised Clustering
| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmm_2 | 0.0271 | -0.0083 | 0.0099 | 0.6583 | 1 | 2 | 2 |
| kmeans_2 | 0.0000 | 0.0229 | 0.0132 | 0.5917 | 81 | 48 | 48 |
| agglomerative_2 | 0.0354 | -0.0211 | 0.0030 | 0.5750 | 15 | 20 | 27 |
- Null near-80 rate: `0.0000` (n=5000)

## Check 2: Name-Masked + Topic-Suppressed Embedding Unsupervised Clustering
- Embedding model: `shibing624/text2vec-base-chinese-paraphrase`
- Topic terms removed: `400`
| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| agglomerative_2 | 0.0458 | 0.1494 | 0.0790 | 0.7083 | 101 | 30 | 38 |
| kmeans_2 | 0.0685 | 0.0508 | 0.1021 | 0.6250 | 45 | 34 | 41 |
| gmm_2 | 0.0714 | 0.0236 | 0.0960 | 0.6000 | 45 | 33 | 41 |
- Null near-80 rate: `0.0000` (n=5000)
- Observed best split: `101` (near 80: `False`)

## Check 3: Baseline E5 Label-Order Null Test
- Baseline best method: `gmm_2`
- Observed best split chapter: `76`
- Null near-80 rate: `0.0266`
- Null near-80 and error<=observed rate: `0.0000`

## Outputs
- Summary JSON: `artifacts/07_contamination_checks/text2vec_zh/summary.json`
- Null histogram figure: `artifacts/07_contamination_checks/text2vec_zh/null_split_hist_masked_embedding.png`