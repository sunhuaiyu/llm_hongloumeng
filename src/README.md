# Source Map

`src/` contains the executable analysis pipeline.

## Pipeline Order
1. `parse_chapters.py`
2. `extract_stylometry.py`
3. `llm_signals_mlx.py`
4. `run_tests.py`
5. `run_ablations.py`
6. `embed_cluster_analysis.py`
7. `contamination_resistant_checks.py`

## Script Roles

### `parse_chapters.py`
- Input: `HongLouMeng.txt`
- Output: `data/chapters/`, `artifacts/01_parse/chapter_qc.csv`

### `extract_stylometry.py`
- Input: `data/chapters/`
- Output: `artifacts/02_stylometry/stylometry_chunk.parquet`, `artifacts/02_stylometry/stylometry_chapter.parquet`

### `llm_signals_mlx.py`
- Input: stylometry chunk table plus prompt templates from `prompts/`
- Output: `artifacts/03_llm_signals/`

### `run_tests.py`
- Input: stylometry and optional LLM feature tables
- Output: report markdown, json summaries, and figures

### `run_ablations.py`
- Input: stylometry chunk table
- Output: `artifacts/05_ablations/`

### `embed_cluster_analysis.py`
- Input: chunk text table
- Output: model-specific folders in `artifacts/06_embedding_clustering/`

### `contamination_resistant_checks.py`
- Input: `data/chapters/`
- Output: model-specific folders in `artifacts/07_contamination_checks/`

## Practical Rule
- Change files in `src/` when behavior should change.
- Do not edit generated outputs in `artifacts/` to simulate a pipeline change.
