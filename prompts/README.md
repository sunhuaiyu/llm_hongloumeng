# Prompt Map

`prompts/` contains templates used by `src/llm_signals_mlx.py`.

## Files
- `literary_signs_system.txt`
- `literary_signs_user_template.txt`
- `literary_signs_system_fast.txt`
- `literary_signs_user_template_fast.txt`

## Purpose
These prompts define how the local MLX model scores literary signals for each chunk.

## Practical Rule
- If LLM-derived signal quality changes, inspect these files together with `src/llm_signals_mlx.py`.
