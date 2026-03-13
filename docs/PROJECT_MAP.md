# Project Map

## Purpose
This file gives a simple map of the repository and, in particular, explains the new step-based `artifacts/` layout.

## One-Screen Summary
- Source corpus: `HongLouMeng.txt`
- Derived chapter text: `data/chapters/`
- Analysis code: `src/`
- Prompt templates: `prompts/`
- Human docs: `docs/`
- Generated outputs: `artifacts/`

## What Has Been Implemented
Three analysis tracks have already been built and run:

1. Supervised stylometry plus MLX literary-signal testing
2. Unsupervised embedding clustering across multiple models
3. Contamination-resistant unsupervised checks

## New Artifact Structure
The old `features/`, `figures/`, and `reports/` split was not good for answering "which step wrote this file?"

The new structure is step-based:

```text
artifacts/
├── 00_overview/                # cross-track summaries
├── 01_parse/                   # parse_chapters.py
├── 02_stylometry/              # extract_stylometry.py
├── 03_llm_signals/             # llm_signals_mlx.py
├── 04_authorship_tests/        # run_tests.py
├── 05_ablations/               # run_ablations.py
├── 06_embedding_clustering/    # embed_cluster_analysis.py
├── 07_contamination_checks/    # contamination_resistant_checks.py
└── 99_logs/                    # logs
```

## Reading Order
If you want to understand the project quickly, read in this order:

1. `README.md`
2. `docs/PROJECT_MAP.md`
3. `TASK.md`
4. `artifacts/00_overview/RESULTS_SUMMARY.md`
5. `artifacts/README.md`
6. `src/README.md`

## Flow Of Work
```text
HongLouMeng.txt
  -> src/parse_chapters.py
  -> artifacts/01_parse/
  -> data/chapters/

data/chapters/
  -> src/extract_stylometry.py
  -> artifacts/02_stylometry/

artifacts/02_stylometry/
  -> src/llm_signals_mlx.py
  -> artifacts/03_llm_signals/

artifacts/02_stylometry/ + artifacts/03_llm_signals/
  -> src/run_tests.py
  -> artifacts/04_authorship_tests/

artifacts/02_stylometry/
  -> src/run_ablations.py
  -> artifacts/05_ablations/

artifacts/02_stylometry/
  -> src/embed_cluster_analysis.py
  -> artifacts/06_embedding_clustering/

data/chapters/
  -> src/contamination_resistant_checks.py
  -> artifacts/07_contamination_checks/
```

## Most Important Existing Outputs
- Top-level summary: `artifacts/00_overview/RESULTS_SUMMARY.md`
- Default supervised test run: `artifacts/04_authorship_tests/default/`
- Full 1171-chunk Qwen 0.5B run: `artifacts/04_authorship_tests/full1171_q05/`
- Ablation outputs: `artifacts/05_ablations/`
- Embedding comparisons: `artifacts/06_embedding_clustering/comparisons/`
- Contamination-check comparisons: `artifacts/07_contamination_checks/comparisons/`

## Practical Rule
When browsing `artifacts/`, first ask "which pipeline step do I care about?" and enter the numbered stage directory for that step.
