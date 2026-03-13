# Methods (Full1171 Qwen2.5-0.5B Run)

## Inputs
- Text: `HongLouMeng.txt`
- Parsed chapters: `data/chapters/`
- Stylometry features:
  - `artifacts/02_stylometry/stylometry_chunk.parquet` (1171 chunks)
  - `artifacts/02_stylometry/stylometry_chapter.parquet` (120 chapters)

## LLM Extraction
- Script: `src/llm_signals_mlx.py`
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Prompts:
  - `prompts/literary_signs_system_fast.txt`
  - `prompts/literary_signs_user_template_fast.txt`
- Key params:
  - `repeats=1`
  - `max_tokens=140`
  - `parse_retries=0`
  - `checkpoint_every=50`
  - `resume=true`
- Outputs:
  - `artifacts/03_llm_signals/runs/full1171_q05/llm_signals_chunk.parquet` (1171 rows)
  - `artifacts/03_llm_signals/runs/full1171_q05/llm_signals_chapter.parquet` (120 rows)

## Statistical Testing
- Script: `src/run_tests.py`
- Permutations: `200`
- Inputs:
  - stylometry: `artifacts/02_stylometry/stylometry_chunk.parquet`, `artifacts/02_stylometry/stylometry_chapter.parquet`
  - llm: `artifacts/03_llm_signals/runs/full1171_q05/llm_signals_chunk.parquet`
- Outputs:
  - `artifacts/04_authorship_tests/full1171_q05/results.json`
  - `artifacts/04_authorship_tests/full1171_q05/main_results.md`
  - `artifacts/04_authorship_tests/full1171_q05/figures/chapter_change_points.png`
  - `artifacts/04_authorship_tests/full1171_q05/figures/stylometry_permutation_auc.png`
