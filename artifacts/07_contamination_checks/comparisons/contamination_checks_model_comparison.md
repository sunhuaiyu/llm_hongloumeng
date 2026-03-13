# Contamination-Resistant Model Comparison

All runs used: name-masking + top-400 topic-term suppression + 5000 null permutations.

| Run | Model | Best Method | Best-Flip Acc | Best Split | Near-80 | Split Error | Transitions | Null Near-80 Rate | Null Err<=Obs | Null Near-80 & Err<=Obs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| e5_small | intfloat/multilingual-e5-small | gmm_2 | 0.6833 | 74 | False | 36 | 39 | 0.0376 | 0.0004 | 0.0000 |
| bge_small_zh | BAAI/bge-small-zh-v1.5 | agglomerative_2 | 0.6750 | 19 | False | 34 | 35 | 0.0136 | 0.0008 | 0.0000 |
| text2vec_zh | shibing624/text2vec-base-chinese-paraphrase | agglomerative_2 | 0.7083 | 101 | False | 30 | 38 | 0.0000 | 0.0836 | 0.0000 |

## Notes
- `null_near_80_rate` is the chance a shuffled label order yields a split within chapter 80±5.
- `null_near_80_and_err_le_observed_rate` is stricter: near-80 and at least as good fit as observed.