# Methods Used in This Run

## Environment
- Python: project-local `venv/` (Python 3.12.12)
- Hardware target: Apple Silicon Mac
- LLM runtime: `mlx-lm`
- LLM model: `mlx-community/Qwen2.5-3B-Instruct-4bit`

## Inputs
- Novel text: `HongLouMeng.txt`
- Chapter parsing output: `data/chapters/chapter_001.txt` ... `data/chapters/chapter_120.txt`
- Chapter QC: `artifacts/reports/chapter_qc.csv`

## Feature Extraction
- Stylometric chunks: `src/extract_stylometry.py`
  - chunk size: 1400 chars
  - chunk stride: 700 chars
  - min chunk chars: 450
  - ngram hash features: 512
- Outputs:
  - `artifacts/features/stylometry_chunk.parquet` (1171 rows)
  - `artifacts/features/stylometry_chapter.parquet` (120 rows)

## LLM Literary Signals
- Script: `src/llm_signals_mlx.py`
- Prompt files:
  - `prompts/literary_signs_system.txt`
  - `prompts/literary_signs_user_template.txt`
- This completed run used one representative chunk per chapter:
  - Input: `artifacts/features/chapter120/stylometry_chunk_one_per_chapter.parquet` (120 rows)
  - repeats: 1
  - max_tokens: 220
  - parse_retries: 1
  - checkpoint_every: 10
- Outputs:
  - `artifacts/features/chapter120/llm_signals_chunk.parquet` (120 rows)
  - `artifacts/features/chapter120/llm_signals_chapter.parquet` (120 rows)

## Main Statistical Test
- Script: `src/run_tests.py`
- Inputs:
  - stylometry chunk/chapter from `artifacts/features/`
  - llm chunk from `artifacts/features/chapter120/llm_signals_chunk.parquet`
- permutations: 200
- Outputs:
  - `artifacts/reports/results.json`
  - `artifacts/reports/main_results.md`
  - `artifacts/figures/chapter_change_points.png`
  - `artifacts/figures/stylometry_permutation_auc.png`

## Robustness Ablations
- Script: `src/run_ablations.py`
- permutations per ablation: 100
- Ablations executed:
  - baseline
  - narration_dominant
  - exclude_short_chunks
  - exclude_verse_heavy
  - name_masked
- Outputs:
  - `artifacts/reports/ablations/ablation_summary.json`
  - `artifacts/reports/ablations/ablation_summary.md`
  - per-ablation feature and result files in `artifacts/reports/ablations/`
