# Artifacts Map

`artifacts/` is now organized by pipeline step so you can tell which script produced which files.

## Stage Map
- `00_overview/`: cross-track summaries for humans
- `01_parse/`: outputs from `src/parse_chapters.py`
- `02_stylometry/`: outputs from `src/extract_stylometry.py`
- `03_llm_signals/`: outputs from `src/llm_signals_mlx.py`
- `04_authorship_tests/`: outputs from `src/run_tests.py`
- `05_ablations/`: outputs from `src/run_ablations.py`
- `06_embedding_clustering/`: outputs from `src/embed_cluster_analysis.py`
- `07_contamination_checks/`: outputs from `src/contamination_resistant_checks.py`
- `99_logs/`: logs and run traces

## What Lives Where

### `00_overview/`
- `RESULTS_SUMMARY.md`
- cross-track synthesis, not a raw pipeline step

### `01_parse/`
- `chapter_qc.csv`

### `02_stylometry/`
- `stylometry_chunk.parquet`
- `stylometry_chapter.parquet`
- `samples/`
- `experiments/`

### `03_llm_signals/`
- `default/`: default MLX signal extraction outputs
- `runs/`: named or historical MLX runs such as `chapter120`, `full1171`, `full1171_q05`

### `04_authorship_tests/`
- `default/`: default supervised test outputs and figures
- `full1171_q05/`: full-corpus Qwen 0.5B run outputs

### `05_ablations/`
- ablation summaries
- per-ablation feature tables
- per-ablation result JSON files

### `06_embedding_clustering/`
- one folder per embedding model
- `comparisons/` for cross-model summaries

### `07_contamination_checks/`
- one folder per contamination-resistant run family or model
- `comparisons/` for cross-model summaries

## Fast Navigation
If you want the quickest reading path:

1. `00_overview/RESULTS_SUMMARY.md`
2. `04_authorship_tests/default/`
3. `05_ablations/`
4. `06_embedding_clustering/comparisons/`
5. `07_contamination_checks/comparisons/`
