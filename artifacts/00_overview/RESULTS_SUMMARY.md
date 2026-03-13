# Results Summary

## Scope
This summary consolidates:
- full-corpus supervised boundary testing (stylometry + LLM literary signals),
- baseline embedding-space unsupervised clustering,
- contamination-resistant unsupervised checks.

Corpus: `HongLouMeng.txt` (120 chapters)  
Chunk basis for full runs: 1171 chunks

## A. Supervised Boundary Testing (Full 1171-Chunk Run)

Primary result file:
- `artifacts/04_authorship_tests/full1171_q05/results.json`

Primary figures:
- `artifacts/04_authorship_tests/full1171_q05/figures/chapter_change_points.png`
- `artifacts/04_authorship_tests/full1171_q05/figures/stylometry_permutation_auc.png`

Model/runtime for literary-signal pass:
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

Key metrics (200 permutations):
- Decision: `partial_support`
- Detected BinSeg change-point: chapter `5`
- Boundary near chapter 80 (+/-5): `False`

Stylometry:
- AUC mean: `0.9782`
- Balanced accuracy mean: `0.9158`
- Permutation p-value: `0.004975`

LLM literary signals:
- AUC mean: `0.5311`
- Balanced accuracy mean: `0.5000`
- Permutation p-value: `0.164179` (not significant)

Fused (stylometry + LLM):
- AUC mean: `0.9784`
- Balanced accuracy mean: `0.9204`
- Permutation p-value: `0.004975`

Interpretation:
- Strong supervised separability exists in stylometric space.
- LLM literary-score features alone are weak in this full 0.5B run.
- The strongest detected break is not around chapter 80.

## B. Robustness Ablations (Stylometry Track)

Summary file:
- `artifacts/05_ablations/ablation_summary.md`

Highlights:
- All tested ablations retained significant separability.
- Best split chapters remained far from 80 (e.g., 5 or 15 in reported runs).
- This supports a stable style signal, but not the specific 80/40 boundary location.

## C. Baseline Embedding-Space Unsupervised Clustering

Per-model outputs:
- `artifacts/06_embedding_clustering/bge_small_zh/report.md`
- `artifacts/06_embedding_clustering/e5_small/report.md`
- `artifacts/06_embedding_clustering/text2vec_zh/report.md`

Cross-model comparison:
- `artifacts/06_embedding_clustering/comparisons/embedding_cluster_comparison.md`
- `artifacts/06_embedding_clustering/comparisons/embedding_cluster_comparison.csv`

### C1. `BAAI/bge-small-zh-v1.5`
Best method: `gmm_2`
- Best-flip accuracy: `0.5583`
- ARI: `0.0048`
- NMI: `0.0052`
- Best split chapter: `18`

Interpretation:
- Weak unsupervised alignment to 1-80 vs 81-120.

### C2. `intfloat/multilingual-e5-small`
Best method: `gmm_2`
- Best-flip accuracy: `0.8333`
- ARI: `0.4379`
- NMI: `0.3405`
- Best split chapter: `76`

Interpretation:
- Embedding space shows a substantially clearer unsupervised two-group structure.
- The best split is near (but not exactly at) chapter 80.

### C3. `shibing624/text2vec-base-chinese-paraphrase`
Best method: `agglomerative_2`
- Best-flip accuracy: `0.5500`
- ARI: `-0.0016`
- NMI: `0.0256`
- Best split chapter: `38`

Interpretation:
- Weak unsupervised alignment to 1-80 vs 81-120.

## D. Contamination-Resistant Unsupervised Checks

Primary outputs:
- `artifacts/07_contamination_checks/e5_small/report.md`
- `artifacts/07_contamination_checks/bge_small_zh/report.md`
- `artifacts/07_contamination_checks/text2vec_zh/report.md`
- `artifacts/07_contamination_checks/comparisons/contamination_checks_model_comparison.md`

### D1. Corpus-Only Check (No Embedding Model)
From `artifacts/07_contamination_checks/e5_small/report.md`:
- TF-IDF char-ngram best method: `gmm_2`
- Best split chapter: `1`
- Null near-80 rate: `0.0000` (5000 permutations)

Interpretation:
- Without external embeddings, no near-80 boundary signal appears.

### D2. Name-Masked + Topic-Suppressed Embedding Checks (Cross-Model)
All runs use:
- character-name masking,
- removal of top 400 frequent CJK topic terms,
- 5000 chapter-order null permutations.

Results:
- `intfloat/multilingual-e5-small`:
  - best method: `gmm_2`
  - best-flip accuracy: `0.6833`
  - best split chapter: `74` (near-80: `False`)
  - null near-80 rate: `0.0376`
  - null near-80 and error<=observed: `0.0000`
- `BAAI/bge-small-zh-v1.5`:
  - best method: `agglomerative_2`
  - best-flip accuracy: `0.6750`
  - best split chapter: `19` (near-80: `False`)
  - null near-80 rate: `0.0136`
  - null near-80 and error<=observed: `0.0000`
- `shibing624/text2vec-base-chinese-paraphrase`:
  - best method: `agglomerative_2`
  - best-flip accuracy: `0.7083`
  - best split chapter: `101` (near-80: `False`)
  - null near-80 rate: `0.0000`
  - null near-80 and error<=observed: `0.0000`

Interpretation:
- The near-80 split is not robust after contamination-resistant preprocessing.
- Cross-model splits diverge materially (19, 74, 101), indicating sensitivity and lower boundary stability.

## E. Overall Interpretation

1. The pipeline consistently detects strong stylistic structure in the corpus.
2. In supervised tests, the strongest change-point is not near chapter 80.
3. Baseline unsupervised results are model-sensitive, with only baseline `multilingual-e5-small` showing a near-80 split.
4. Contamination-resistant checks reduce or remove the near-80 pattern across all tested embedding models.
5. Current evidence supports stylistic heterogeneity in the novel, but does not provide robust support for an exact 80/40 authorship boundary.

## F. Recommended Next Validation

1. Add a classical-Chinese-specialized embedding model and repeat the same contamination-resistant suite.
2. Run boundary-window bootstrap analysis across chapters 60-100 to quantify split uncertainty.
3. Add function-word/POS-dominant unsupervised features to further reduce topic confounds.

## G. Plain-Language "So What" Summary

1. The study used multiple methods so conclusions do not depend on one tool.
2. The writing style clearly changes in the novel, so the text is not stylistically uniform.
3. The strongest break found by supervised tests is near the start of the book, not around chapter 80.
4. The small local LLM literary features were weak on their own for finding a reliable boundary.
5. Most strong evidence comes from stylometry rather than LLM literary scoring.
6. Stress tests (ablations) still showed style differences, so those differences are likely real.
7. Those stress tests still did not place the main break near chapter 80.
8. Unsupervised grouping gives different answers across embedding models.
9. One baseline model suggested a near-80 split, but other baseline models did not.
10. After stricter anti-bias preprocessing (name masking and topic suppression), all tested models moved away from a near-80 split.
11. A model-free corpus-only TF-IDF check also did not support a near-80 boundary.
12. Current evidence supports stylistic heterogeneity, but not a robust exact 80/40 authorship boundary.
13. The next best validation is a classical-Chinese-specialized model plus stronger topic-resistant features.
