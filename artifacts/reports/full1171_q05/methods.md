# Methods (Full1171 Qwen2.5-0.5B Run)

## Inputs
- Text: `HongLouMeng.txt`
- Parsed chapters: `data/chapters/`
- Stylometry features:
  - `artifacts/features/stylometry_chunk.parquet` (1171 chunks)
  - `artifacts/features/stylometry_chapter.parquet` (120 chapters)

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
  - `artifacts/features/full1171_q05/llm_signals_chunk.parquet` (1171 rows)
  - `artifacts/features/full1171_q05/llm_signals_chapter.parquet` (120 rows)

## Statistical Testing
- Script: `src/run_tests.py`
- Permutations: `200`
- Inputs:
  - stylometry: `artifacts/features/stylometry_chunk.parquet`, `artifacts/features/stylometry_chapter.parquet`
  - llm: `artifacts/features/full1171_q05/llm_signals_chunk.parquet`
- Outputs:
  - `artifacts/reports/full1171_q05/results.json`
  - `artifacts/reports/full1171_q05/main_results.md`
  - `artifacts/figures/full1171_q05/chapter_change_points.png`
  - `artifacts/figures/full1171_q05/stylometry_permutation_auc.png`
