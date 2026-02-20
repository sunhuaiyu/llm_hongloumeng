# llm_hongloumeng

Local authorship-analysis workflow for `HongLouMeng.txt` (120 chapters), focused on testing whether chapters 81-120 differ from chapters 1-80.

## What this repo includes
- Chapter parsing and preprocessing
- Stylometric feature extraction
- Local MLX-based LLM literary-signal extraction
- Supervised boundary tests and ablations
- Unsupervised embedding clustering
- Contamination-resistant checks (name masking, topic suppression, null tests)

## Quick start
```bash
python3.12 -m venv venv
source venv/bin/activate
```

Main code is in `src/`.  
Primary reports are in `artifacts/reports/`.
