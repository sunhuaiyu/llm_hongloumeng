# AGENTS.md

## Scope
These instructions apply to the entire repository rooted here.

## Repo Purpose
This repo runs a local authorship-analysis workflow for `HongLouMeng.txt`, focused on whether chapters 81-120 differ from chapters 1-80.

Primary code lives in `src/`.
Primary generated outputs live in `artifacts/`.
Prompts for MLX-based scoring live in `prompts/`.
Raw parsed chapter text lives in `data/chapters/`.
Human-facing navigation docs live in `docs/`.

## Working Rules
- Use the project-local virtual environment at `venv/`.
- Prefer `venv/bin/python` over a system Python when running scripts.
- Treat `artifacts/` as generated output. Do not hand-edit files there unless the user explicitly asks for report text edits.
- Treat `data/chapters/` as derived from `HongLouMeng.txt`. If parsing logic changes, regenerate them with `src/parse_chapters.py`.
- Preserve existing run directories and model-specific outputs unless the task requires regeneration.
- Only modify `AGENTS.md` when the user has explicitly approved that specific `AGENTS.md` change.
- When changing analysis behavior, edit code in `src/` or prompt files in `prompts/`, then rerun the smallest necessary downstream step.
- For quick repository orientation, read `docs/PROJECT_MAP.md` before diving into model-specific artifact folders.

## Directory Structure Pattern
Use a role-based top level and a step-based `artifacts/` tree.

Top-level expectations:
- `src/`: executable pipeline code
- `data/`: derived corpus data used as pipeline input
- `prompts/`: prompt templates only
- `docs/`: human-facing navigation and project docs
- `artifacts/`: generated outputs only

`artifacts/` expectations:
- Use numbered stage directories so the producing step is obvious.
- Keep stage numbers stable once introduced.
- Prefer `00_*` for overview material and `99_*` for logs or misc run traces.
- Put outputs from one script under one stage directory, not split across separate `features/`, `figures`, and `reports` roots.

Current canonical stage layout:
- `artifacts/00_overview/`
- `artifacts/01_parse/`
- `artifacts/02_stylometry/`
- `artifacts/03_llm_signals/`
- `artifacts/04_authorship_tests/`
- `artifacts/05_ablations/`
- `artifacts/06_embedding_clustering/`
- `artifacts/07_contamination_checks/`
- `artifacts/99_logs/`

Subdirectory rules inside a stage:
- Use `default/` for the main default run of that stage.
- Use `runs/<run_name>/` for named historical or parameter-specific runs when a stage can have multiple runs.
- Use one folder per model for model-comparison stages, for example `e5_small/` or `bge_small_zh/`.
- Use `comparisons/` for cross-run or cross-model summary files.
- Keep figures with the run that produced them, for example `artifacts/04_authorship_tests/full1171_q05/figures/`.

Naming rules:
- Prefer short lowercase snake_case directory names.
- Name directories by pipeline role first, then run or model.
- Avoid top-level artifact folders named only by implementation history or ad hoc experiments.
- When adding a new persistent analysis stage, assign the next stage number instead of overloading an unrelated folder.

## Canonical Pipeline
Run steps in this order when a full rebuild is needed:

```bash
venv/bin/python src/parse_chapters.py
venv/bin/python src/extract_stylometry.py
venv/bin/python src/llm_signals_mlx.py
venv/bin/python src/run_tests.py
venv/bin/python src/run_ablations.py
venv/bin/python src/embed_cluster_analysis.py
venv/bin/python src/contamination_resistant_checks.py
```

## Important Scripts
- `src/parse_chapters.py`: parses `HongLouMeng.txt` into `data/chapters/` and writes QC metadata to `artifacts/01_parse/`.
- `src/extract_stylometry.py`: builds chunk-level and chapter-level stylometric features in `artifacts/02_stylometry/`.
- `src/llm_signals_mlx.py`: scores chunks with a local MLX model using prompt templates in `prompts/`.
- `src/run_tests.py`: runs grouped classification, permutation tests, and change-point analysis.
- `src/run_ablations.py`: runs stylometry robustness ablations and writes summaries under `artifacts/05_ablations/`.
- `src/embed_cluster_analysis.py`: runs unsupervised embedding clustering for a chosen embedding model.
- `src/contamination_resistant_checks.py`: runs TF-IDF, masked-name, topic-suppressed, and null-test checks.

## Canonical Existing Outputs
- Project summary: `artifacts/00_overview/`
- Chapter QC: `artifacts/01_parse/chapter_qc.csv`
- Stylometry features: `artifacts/02_stylometry/`
- LLM signal runs: `artifacts/03_llm_signals/`
- Main supervised tests: `artifacts/04_authorship_tests/`
- Ablations: `artifacts/05_ablations/`
- Embedding clustering: `artifacts/06_embedding_clustering/`
- Contamination checks: `artifacts/07_contamination_checks/`

## Environment Notes
- The repo is designed for local macOS execution on Apple Silicon.
- MLX-based runs can be slow and may depend on model availability in the local environment.
- Embedding scripts may download or load sentence-transformer models if they are not already cached.

## Change Guidance
- If the user asks for methodological changes, update `TASK.md` only if the plan itself has changed materially.
- If the user asks for prompt-quality or LLM-signal changes, inspect both the prompt templates and `src/llm_signals_mlx.py`.
- If a task only affects reporting, prefer changing the producing script rather than editing generated markdown/json directly.
- When reporting results, state clearly which output directory and model variant produced them.
- If a task introduces new outputs, place them in the existing stage directory that matches the producing script; only create a new stage directory when the output belongs to a genuinely new pipeline step.
- If a requested `AGENTS.md` change is ambiguous, default to no `AGENTS.md` edit until the user approves the specific modification.
