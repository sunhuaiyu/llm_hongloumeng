#!/usr/bin/env python3
"""Extract chapter/chunk literary signals using a local MLX model."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

METRICS = [
    "narrative_distance",
    "irony_intensity",
    "descriptive_density",
    "dialogue_naturalness",
    "character_interiority",
    "transition_smoothness",
    "emotional_tonality",
    "closure_foreshadowing_style",
]


def require_dependencies():
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "source venv/bin/activate && pip install -U pandas numpy pyarrow"
        ) from exc

    try:
        from mlx_lm import generate, load  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing MLX runtime. Install with:\n"
            "source venv/bin/activate && pip install -U mlx-lm"
        ) from exc


def load_table(path: Path):
    import pandas as pd

    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)

    if path.suffix == ".parquet":
        csv_alt = path.with_suffix(".csv")
        if csv_alt.exists():
            return pd.read_csv(csv_alt)
    elif path.suffix == ".csv":
        pq_alt = path.with_suffix(".parquet")
        if pq_alt.exists():
            return pd.read_parquet(pq_alt)

    raise FileNotFoundError(f"Could not find {path} or fallback csv/parquet pair")


def save_table(df, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    pq_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    try:
        df.to_parquet(pq_path, index=False)
        return pq_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        "System:\n"
        f"{system_prompt}\n\n"
        "User:\n"
        f"{user_prompt}\n\n"
        "Assistant:\n"
    )


def _call_generate(generate_fn, model, tokenizer, prompt: str, max_tokens: int, temp: float):
    candidates = [
        {"max_tokens": max_tokens, "temp": temp, "verbose": False},
        {"max_tokens": max_tokens, "temperature": temp, "verbose": False},
        {"max_tokens": max_tokens, "verbose": False},
    ]
    last_exc = None
    for kwargs in candidates:
        try:
            return generate_fn(model, tokenizer, prompt=prompt, **kwargs)
        except TypeError as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError("MLX generate call failed without a captured exception")


def parse_llm_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    candidate = text
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            payload = None
        else:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    else:
        payload = payload

    if isinstance(payload, dict):
        scores = payload.get("scores")
        if isinstance(scores, dict):
            parsed = {}
            for metric in METRICS:
                value = scores.get(metric)
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    parsed = {}
                    break
                parsed[metric] = min(max(v, 0.0), 5.0)

            if parsed:
                evidence = payload.get("evidence", [])
                if isinstance(evidence, list):
                    parsed["evidence"] = [str(x) for x in evidence[:3]]
                else:
                    parsed["evidence"] = []
                return parsed

    # Fallback: recover scores from malformed/truncated JSON-like text.
    recovered = {}
    for metric in METRICS:
        m = re.search(rf"\"{re.escape(metric)}\"\s*:\s*(-?\d+(?:\.\d+)?)", text)
        if not m:
            return None
        try:
            v = float(m.group(1))
        except (TypeError, ValueError):
            return None
        recovered[metric] = min(max(v, 0.0), 5.0)
    recovered["evidence"] = []
    return recovered


def nanmean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return (sum(vals) / len(vals)) if vals else float("nan")


def nanvar(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    mu = sum(vals) / len(vals)
    return sum((v - mu) ** 2 for v in vals) / len(vals)


def main() -> None:
    require_dependencies()

    import numpy as np
    import pandas as pd
    from mlx_lm import generate, load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("artifacts/02_stylometry/stylometry_chunk.parquet"),
        help="Chunk feature table from extract_stylometry.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/03_llm_signals/default"),
        help="Output directory.",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="MLX model identifier.",
    )
    parser.add_argument("--system-prompt", type=Path, default=Path("prompts/literary_signs_system.txt"))
    parser.add_argument(
        "--user-template",
        type=Path,
        default=Path("prompts/literary_signs_user_template.txt"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=1,
        help="Extra generation attempts per run when JSON parsing fails.",
    )
    parser.add_argument(
        "--retry-max-tokens",
        type=int,
        default=640,
        help="Upper bound for max_tokens when retrying parse failures.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write intermediate outputs every N processed chunks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing llm_signals_chunk output if present.",
    )
    parser.add_argument(
        "--resume-retry-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, retry rows with llm_parse_success_rate <= 0.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N processed chunks.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    args = parser.parse_args()

    chunk_df = load_table(args.chunks)
    required_cols = {"chapter_id", "chunk_id", "chunk_text", "period_label"}
    missing = required_cols - set(chunk_df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in chunk table: {sorted(missing)}")

    if args.limit and args.limit > 0:
        chunk_df = chunk_df.head(args.limit).copy()

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    user_template = args.user_template.read_text(encoding="utf-8").strip()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    out_chunk_base = args.out_dir / "llm_signals_chunk"
    out_chapter_base = args.out_dir / "llm_signals_chapter"

    records = []
    processed_chunk_ids: set[str] = set()
    if args.resume:
        existing = None
        for p in [out_chunk_base.with_suffix(".parquet"), out_chunk_base.with_suffix(".csv")]:
            if p.exists():
                existing = load_table(p)
                break
        if existing is not None and not existing.empty:
            if "chunk_id" in existing.columns:
                existing["chunk_id"] = existing["chunk_id"].astype(str)
                if args.resume_retry_failures and "llm_parse_success_rate" in existing.columns:
                    success_mask = existing["llm_parse_success_rate"].fillna(0.0) > 0.0
                    retry_count = int((~success_mask).sum())
                    kept = existing[success_mask].copy()
                    processed_chunk_ids = set(kept["chunk_id"].tolist())
                    records = kept.to_dict(orient="records")
                    print(
                        f"Resuming from {len(processed_chunk_ids)} successful rows in {p}; "
                        f"retrying {retry_count} failed rows."
                    )
                else:
                    processed_chunk_ids = set(existing["chunk_id"].tolist())
                    records = existing.to_dict(orient="records")
                    print(
                        f"Resuming from {len(processed_chunk_ids)} existing chunk rows in {p}."
                    )
            else:
                print("Resume requested, but existing output lacks chunk_id. Ignoring resume state.")

    if processed_chunk_ids:
        chunk_df = chunk_df[~chunk_df["chunk_id"].astype(str).isin(processed_chunk_ids)].copy()
        print(f"Chunks left to process after resume filter: {len(chunk_df)}")
    total_to_process = len(chunk_df)
    for idx, row in enumerate(chunk_df.itertuples(index=False), start=1):
        chapter_id = int(row.chapter_id)
        chunk_id = str(row.chunk_id)
        chunk_text = str(row.chunk_text)
        period_label = int(row.period_label)

        parsed_runs = []
        raw_runs = []
        for run_idx in range(1, args.repeats + 1):
            user_prompt = user_template.format(
                chapter_id=chapter_id,
                chunk_id=chunk_id,
                run_index=run_idx,
                text=chunk_text,
            )
            prompt = build_prompt(tokenizer, system_prompt, user_prompt)
            current_max_tokens = args.max_tokens
            parsed = None
            raw = ""
            for _ in range(args.parse_retries + 1):
                raw = _call_generate(
                    generate_fn=generate,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=current_max_tokens,
                    temp=args.temperature,
                )
                raw = str(raw).strip()
                parsed = parse_llm_json(raw)
                if parsed is not None:
                    break
                current_max_tokens = min(current_max_tokens * 2, args.retry_max_tokens)

            raw_runs.append(raw[:1000])
            if parsed is not None:
                parsed_runs.append(parsed)

        record = {
            "chapter_id": chapter_id,
            "chunk_id": chunk_id,
            "period_label": period_label,
            "llm_model": args.model,
            "llm_parse_success_rate": len(parsed_runs) / args.repeats,
            "llm_evidence": " | ".join(parsed_runs[0]["evidence"]) if parsed_runs else "",
            "llm_raw_sample": raw_runs[0] if raw_runs else "",
        }

        for metric in METRICS:
            vals = [run[metric] for run in parsed_runs] if parsed_runs else [float("nan")]
            record[f"llm_{metric}_mean"] = float(nanmean(vals))
            record[f"llm_{metric}_var"] = float(nanvar(vals))
        records.append(record)

        if args.log_every > 0 and (idx % args.log_every == 0 or idx == 1 or idx == total_to_process):
            print(
                f"Processed chunk {idx}/{total_to_process} "
                f"(chapter={chapter_id}, chunk_id={chunk_id}, parse_rate={record['llm_parse_success_rate']:.2f})"
            )
        if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
            checkpoint_df = pd.DataFrame(records)
            out_path = save_table(checkpoint_df, out_chunk_base)
            print(f"Checkpoint saved after {idx} processed chunks -> {out_path}")

    llm_chunk_df = pd.DataFrame(records)
    llm_metric_cols = [c for c in llm_chunk_df.columns if c.startswith("llm_") and c.endswith("_mean")]

    llm_chapter_df = (
        llm_chunk_df.groupby("chapter_id", as_index=False)[llm_metric_cols + ["llm_parse_success_rate"]]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    llm_chapter_df["period_label"] = (llm_chapter_df["chapter_id"] > 80).astype(int)
    chunk_counts = llm_chunk_df.groupby("chapter_id").size().rename("chunk_count").reset_index()
    llm_chapter_df = llm_chapter_df.merge(chunk_counts, on="chapter_id", how="left")
    llm_chapter_df["llm_model"] = args.model

    out_chunk = save_table(llm_chunk_df, out_chunk_base)
    out_chapter = save_table(llm_chapter_df, out_chapter_base)

    print(f"LLM chunk rows: {len(llm_chunk_df)}")
    print(f"LLM chapter rows: {len(llm_chapter_df)}")
    print(f"Saved chunk signals: {out_chunk}")
    print(f"Saved chapter signals: {out_chapter}")


if __name__ == "__main__":
    main()
