# llm_hongloumeng

Local authorship-analysis workflow for `HongLouMeng.txt`, focused on whether chapters 81-120 differ from chapters 1-80.

## Start Here
- Read `docs/PROJECT_MAP.md` for the simplest explanation of the repo and the new artifact layout.
- Read `TASK.md` for the research plan, hypotheses, and canonical runs.
- Read `artifacts/00_overview/RESULTS_SUMMARY.md` for the top-level findings.

## Top-Level Mental Model
- `HongLouMeng.txt`: source corpus
- `src/`: executable pipeline
- `data/`: parsed chapter text derived from the corpus
- `prompts/`: prompt templates for MLX literary-signal extraction
- `docs/`: human-facing navigation docs
- `artifacts/`: generated outputs, now organized by pipeline step

## Directory Map
```text
.
├── HongLouMeng.txt
├── TASK.md
├── docs/
├── src/
├── data/
├── prompts/
├── artifacts/
│   ├── 00_overview/
│   ├── 01_parse/
│   ├── 02_stylometry/
│   ├── 03_llm_signals/
│   ├── 04_authorship_tests/
│   ├── 05_ablations/
│   ├── 06_embedding_clustering/
│   ├── 07_contamination_checks/
│   └── 99_logs/
├── requirements.txt
└── venv/
```

## Canonical Pipeline
```bash
venv/bin/python src/parse_chapters.py
venv/bin/python src/extract_stylometry.py
venv/bin/python src/llm_signals_mlx.py
venv/bin/python src/run_tests.py
venv/bin/python src/run_ablations.py
venv/bin/python src/embed_cluster_analysis.py
venv/bin/python src/contamination_resistant_checks.py
```

## Where To Look
- For the repo map: `docs/PROJECT_MAP.md`
- For outputs by step: `artifacts/README.md`
- For code entry points: `src/README.md`
- For top-level results: `artifacts/00_overview/RESULTS_SUMMARY.md`
