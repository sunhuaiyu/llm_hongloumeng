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
- AUC mean ± std: `0.5311 ± 0.0233`
- Balanced accuracy mean ± std: `0.5000 ± 0.0000`
- Permutation p-value: `0.164179`

## Fused Features (Stylometry + LLM)
- AUC mean ± std: `0.9784 ± 0.0094`
- Balanced accuracy mean ± std: `0.9204 ± 0.0280`
- Permutation p-value: `0.004975`

## Outputs
- JSON: `artifacts/04_authorship_tests/full1171_q05/results.json`
- Figure: `artifacts/04_authorship_tests/full1171_q05/figures/chapter_change_points.png`
- Figure: `artifacts/04_authorship_tests/full1171_q05/figures/stylometry_permutation_auc.png`