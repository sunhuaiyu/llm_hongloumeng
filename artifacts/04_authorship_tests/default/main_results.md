# Authorship Test Results

- Decision: `partial_support`
- BinSeg change-point chapter: `5`
- PELT change-points: `[]`
- Boundary near 80 (+/-5): `False`

## Stylometry
- AUC mean ± std: `0.9782 ± 0.0095`
- Balanced accuracy mean ± std: `0.9158 ± 0.0285`
- Permutation p-value: `0.004975`

## LLM Literary Signals
- AUC mean ± std: `0.7353 ± 0.1231`
- Balanced accuracy mean ± std: `0.6143 ± 0.1722`
- Permutation p-value: `0.004975`

## Fused Features (Stylometry + LLM)
- AUC mean ± std: `0.9265 ± 0.0379`
- Balanced accuracy mean ± std: `0.7339 ± 0.0453`
- Permutation p-value: `0.004975`

## Outputs
- JSON: `artifacts/04_authorship_tests/default/results.json`
- Figure: `artifacts/04_authorship_tests/default/figures/chapter_change_points.png`
- Figure: `artifacts/04_authorship_tests/default/figures/stylometry_permutation_auc.png`