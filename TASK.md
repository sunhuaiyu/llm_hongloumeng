# PLAN: Authorship Analysis of `HongLouMeng.txt` on Local Mac (Current)

## 1. Goal
Test whether chapters 81-120 were written by a different author than chapters 1-80 using:
- stylometric signals,
- local open-source LLM-derived literary signals,
- unsupervised embedding-space structure,
- contamination-resistant unsupervised checks.

## 2. Constraints and Decisions
- Hardware: Apple M3 Pro, 18 GB unified memory.
- Runtime: fully local on macOS (Apple Silicon friendly).
- Environment: always use project-local `venv/`.
- Corpus: `HongLouMeng.txt`, 120 chapters (`第X章`) confirmed.

Operational decisions implemented:
- Full-corpus literary-signal extraction uses `mlx-community/Qwen2.5-0.5B-Instruct-4bit` for complete 1171-chunk coverage.
- `Qwen2.5-3B` remains optional for higher-capacity subset checks.
- Baseline embedding track uses three open models:
  - `BAAI/bge-small-zh-v1.5`
  - `intfloat/multilingual-e5-small`
  - `shibing624/text2vec-base-chinese-paraphrase`
- Added contamination-resistant track:
  - corpus-only TF-IDF unsupervised clustering,
  - name-masked and topic-suppressed embedding clustering,
  - chapter-order null tests.

## 3. Core Hypotheses
- `H0`: no meaningful authorial boundary between chapters 1-80 and 81-120.
- `H1`: a meaningful boundary exists, strongest near chapter 80.

## 4. Analysis Tracks

### Track A: Stylometry + LLM Supervised Boundary Tests
1. Parse and QC chapters.
2. Chunk text for grouped modeling.
3. Extract stylometric features.
4. Extract literary signals from local MLX model.
5. Run grouped classification + permutation tests + change-point detection.
6. Run stylometry ablations.

### Track B: Embedding-Space Unsupervised Clustering
1. Build chunk embeddings with open-source embedding model.
2. Aggregate to chapter embeddings.
3. Unsupervised clustering in embedding space (`kmeans_2`, `agglomerative_2`, `gmm_2`).
4. Evaluate post hoc alignment to 1-80 vs 81-120 (ARI/NMI/best-flip accuracy/split chapter).
5. Compare models side-by-side.

### Track C: Contamination-Resistant Unsupervised Checks
1. Run corpus-only char-ngram TF-IDF clustering (no external embedding model).
2. Mask major character names, remove top-frequency topic terms, and re-cluster in embedding space.
3. Use chapter-order permutations to quantify how often near-80 splits arise by chance.
4. Compare contamination-resistant outputs across embedding models.

## 5. Implemented Repository Layout
- `src/parse_chapters.py`
- `src/extract_stylometry.py`
- `src/llm_signals_mlx.py`
- `src/run_tests.py`
- `src/run_ablations.py`
- `src/embed_cluster_analysis.py`
- `src/contamination_resistant_checks.py`
- `prompts/literary_signs_system.txt`
- `prompts/literary_signs_user_template.txt`
- `prompts/literary_signs_system_fast.txt`
- `prompts/literary_signs_user_template_fast.txt`
- `data/chapters/`
- `artifacts/00_overview/`
- `artifacts/01_parse/`
- `artifacts/02_stylometry/`
- `artifacts/03_llm_signals/`
- `artifacts/04_authorship_tests/`
- `artifacts/05_ablations/`
- `artifacts/06_embedding_clustering/`
- `artifacts/07_contamination_checks/`

## 6. Current Canonical Run Outputs

### 6.1 Parsing + Stylometry
- Chapters parsed: 120
- Chunk stylometry: `artifacts/02_stylometry/stylometry_chunk.parquet` (1171 rows)
- Chapter stylometry: `artifacts/02_stylometry/stylometry_chapter.parquet` (120 rows)

### 6.2 Full-Corpus LLM Literary Signals
- Script: `src/llm_signals_mlx.py`
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Output:
  - `artifacts/03_llm_signals/runs/full1171_q05/llm_signals_chunk.parquet`
  - `artifacts/03_llm_signals/runs/full1171_q05/llm_signals_chapter.parquet`

### 6.3 Supervised Boundary Testing
- Script: `src/run_tests.py`
- Permutations: 200
- Outputs:
  - `artifacts/04_authorship_tests/full1171_q05/results.json`
  - `artifacts/04_authorship_tests/full1171_q05/main_results.md`
  - `artifacts/04_authorship_tests/full1171_q05/figures/chapter_change_points.png`
  - `artifacts/04_authorship_tests/full1171_q05/figures/stylometry_permutation_auc.png`

### 6.4 Robustness Ablations (Stylometry)
- Script: `src/run_ablations.py`
- Outputs:
  - `artifacts/05_ablations/ablation_summary.json`
  - `artifacts/05_ablations/ablation_summary.md`

### 6.5 Baseline Embedding Unsupervised Clustering
- Script: `src/embed_cluster_analysis.py`
- Output dirs:
  - `artifacts/06_embedding_clustering/bge_small_zh/`
  - `artifacts/06_embedding_clustering/e5_small/`
  - `artifacts/06_embedding_clustering/text2vec_zh/`
- Comparison:
  - `artifacts/06_embedding_clustering/comparisons/embedding_cluster_comparison.md`
  - `artifacts/06_embedding_clustering/comparisons/embedding_cluster_comparison.csv`
  - `artifacts/06_embedding_clustering/comparisons/embedding_cluster_comparison.json`

### 6.6 Contamination-Resistant Cross-Model Checks
- Script: `src/contamination_resistant_checks.py`
- Output dirs:
  - `artifacts/07_contamination_checks/e5_small/`
  - `artifacts/07_contamination_checks/bge_small_zh/`
  - `artifacts/07_contamination_checks/text2vec_zh/`
- Cross-model comparison:
  - `artifacts/07_contamination_checks/comparisons/contamination_checks_model_comparison.md`
  - `artifacts/07_contamination_checks/comparisons/contamination_checks_model_comparison.csv`
  - `artifacts/07_contamination_checks/comparisons/contamination_checks_model_comparison.json`

Key contamination-resistant splits (masked/topic-suppressed):
- `intfloat/multilingual-e5-small`: split 74 (not near 80)
- `BAAI/bge-small-zh-v1.5`: split 19 (not near 80)
- `shibing624/text2vec-base-chinese-paraphrase`: split 101 (not near 80)

## 7. Decision Rule
Support `H1` only when evidence converges across tracks:
- strong break near chapter 80 (for example +/-5),
- significant supervised signal above chance,
- stability under ablations,
- consistency across multiple embedding models,
- persistence under contamination-resistant preprocessing and null tests.

Otherwise report partial support or inconclusive.

## 8. Current Risk Register
- Topic/plot progression confounds style: partially mitigated by masking/topic suppression and ablations.
- LLM scoring noise at small model size: mitigated by strict JSON prompts and parser recovery.
- Embedding-model sensitivity: mitigated by 3-model comparisons.
- Potential pretraining contamination: mitigated by corpus-only TF-IDF and masked/topic-suppressed null-tested runs.
- Overclaim risk: mitigated by permutation/null tests and explicit split-location diagnostics.

## 9. Next-Step Options
1. Add one classical-Chinese-specialized embedding model and run the same contamination-resistant suite.
2. Run local boundary-window stability tests for chapters 60-100 with bootstrap confidence intervals.
3. Add function-word/POS-dominant unsupervised features to further suppress topic leakage.
