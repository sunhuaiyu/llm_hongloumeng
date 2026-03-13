#!/usr/bin/env python3
"""Run robustness ablations on stylometric features and summarize results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def require_dependencies():
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        from sklearn.feature_extraction.text import HashingVectorizer  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "source venv/bin/activate && pip install -U pandas numpy scikit-learn pyarrow"
        ) from exc


def load_table(path: Path):
    import pandas as pd

    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    alt = path.with_suffix(".csv") if path.suffix == ".parquet" else path.with_suffix(".parquet")
    if alt.exists():
        if alt.suffix == ".parquet":
            return pd.read_parquet(alt)
        return pd.read_csv(alt)
    raise FileNotFoundError(f"Missing input table: {path}")


def mask_names(text: str, names: list[str]) -> str:
    masked = text
    for name in names:
        masked = masked.replace(name, "人名")
    return masked


def build_stylometry_from_text(df, text_col: str, n_features: int):
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import HashingVectorizer

    from extract_stylometry import build_base_features

    rows = []
    for row in df.itertuples(index=False):
        text = getattr(row, text_col)
        base = build_base_features(text)
        out = {
            "chapter_id": int(row.chapter_id),
            "chunk_id": str(row.chunk_id),
            "period_label": int(row.period_label),
            "start_char": int(row.start_char),
            "end_char": int(row.end_char),
            "chunk_text": text,
        }
        out.update(base)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        n_features=n_features,
        alternate_sign=False,
        norm=None,
    )
    mat = vectorizer.transform(out_df["chunk_text"].tolist()).astype(np.float32).toarray()
    lens = np.maximum(out_df["base_char_len"].to_numpy(dtype=np.float32), 1.0).reshape(-1, 1)
    mat = mat / lens
    ngram_cols = [f"hgram_{i:04d}" for i in range(n_features)]
    out_df = pd.concat([out_df.reset_index(drop=True), pd.DataFrame(mat, columns=ngram_cols)], axis=1)
    return out_df


def save_table(df, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    pq = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception:
        csv = base_path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def run_single_ablation(name: str, style_df, out_root: Path, n_perm: int, seed: int):
    import numpy as np
    import pandas as pd

    from run_tests import chapter_change_points, chapter_permutation_pvalue, evaluate_grouped_classifier

    if style_df.empty or style_df["chapter_id"].nunique() < 4:
        return {"name": name, "status": "skipped", "reason": "insufficient rows/groups"}

    feature_cols = [c for c in style_df.columns if c.startswith("base_") or c.startswith("hgram_")]
    X = style_df[feature_cols].to_numpy(dtype=float)
    y = style_df["period_label"].to_numpy(dtype=int)
    g = style_df["chapter_id"].to_numpy(dtype=int)

    eval_result = evaluate_grouped_classifier(X, y, g)
    if np.isnan(eval_result.auc_mean):
        return {"name": name, "status": "skipped", "reason": "invalid CV folds"}

    pval, _ = chapter_permutation_pvalue(
        X,
        y,
        g,
        eval_result.auc_mean,
        n_perm,
        seed,
        label=f"ablation:{name}",
        progress_every=10 if n_perm >= 20 else 0,
    )

    chapter_feature_cols = [c for c in feature_cols if c.startswith("base_")]
    chapter_df = (
        style_df.groupby("chapter_id", as_index=False)[chapter_feature_cols]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    ch_ids, _, break_1, pelt_breaks = chapter_change_points(chapter_df, chapter_feature_cols)

    result = {
        "name": name,
        "status": "ok",
        "rows": int(len(style_df)),
        "chapters": int(style_df["chapter_id"].nunique()),
        "auc_mean": float(eval_result.auc_mean),
        "auc_std": float(eval_result.auc_std),
        "bacc_mean": float(eval_result.bacc_mean),
        "bacc_std": float(eval_result.bacc_std),
        "perm_p": float(pval),
        "folds_used": int(eval_result.folds_used),
        "break_binseg": int(break_1),
        "breaks_pelt": pelt_breaks,
        "boundary_near_80": bool(abs(int(break_1) - 80) <= 5),
        "style_significant": bool((not np.isnan(pval)) and pval < 0.05 and eval_result.auc_mean >= 0.60),
    }

    path = out_root / f"{name}_result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["result_path"] = str(path)
    return result


def main() -> None:
    require_dependencies()

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stylometry-chunk",
        type=Path,
        default=Path("artifacts/02_stylometry/stylometry_chunk.parquet"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/05_ablations"))
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dialogue-threshold", type=float, default=0.25)
    parser.add_argument("--min-chars", type=float, default=900.0)
    parser.add_argument("--verse-newline-threshold", type=float, default=0.03)
    args = parser.parse_args()

    style_chunk = load_table(args.stylometry_chunk)
    n_features = len([c for c in style_chunk.columns if c.startswith("hgram_")])
    if n_features == 0:
        raise SystemExit("No hgram_ columns found in stylometry chunk table.")

    base_cols = ["chapter_id", "chunk_id", "period_label", "start_char", "end_char", "chunk_text"]
    source = style_chunk[base_cols + [c for c in style_chunk.columns if c.startswith("base_")]].copy()
    source["chunk_text"] = source["chunk_text"].astype(str)
    source["newline_ratio"] = source["chunk_text"].str.count(r"\n") / source["chunk_text"].str.len().clip(lower=1)

    names = [
        "宝玉",
        "黛玉",
        "宝钗",
        "王熙凤",
        "熙凤",
        "贾母",
        "袭人",
        "晴雯",
        "探春",
        "迎春",
        "惜春",
        "李纨",
        "湘云",
        "妙玉",
        "贾政",
        "贾赦",
        "贾珍",
        "贾琏",
        "贾环",
        "贾兰",
        "凤姐",
        "平儿",
        "紫鹃",
    ]

    ablations = []

    ablations.append(("baseline", source.copy()))
    ablations.append(
        (
            "narration_dominant",
            source[source["base_dialogue_char_ratio"] <= args.dialogue_threshold].copy(),
        )
    )
    ablations.append(("exclude_short_chunks", source[source["base_char_len"] >= args.min_chars].copy()))
    ablations.append(
        (
            "exclude_verse_heavy",
            source[source["newline_ratio"] <= args.verse_newline_threshold].copy(),
        )
    )
    masked = source.copy()
    masked["chunk_text_masked"] = masked["chunk_text"].map(lambda x: mask_names(x, names))
    ablations.append(("name_masked", masked.copy()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, (name, df_variant) in enumerate(ablations):
        print(f"[ablation] running: {name} (rows={len(df_variant)})")
        if name == "name_masked":
            rebuilt = build_stylometry_from_text(df_variant, "chunk_text_masked", n_features=n_features)
        else:
            rebuilt = build_stylometry_from_text(df_variant, "chunk_text", n_features=n_features)

        feat_path = save_table(rebuilt, args.out_dir / f"{name}_stylometry_chunk")
        print(f"[ablation] features saved: {feat_path}")
        result = run_single_ablation(
            name=name,
            style_df=rebuilt,
            out_root=args.out_dir,
            n_perm=args.n_permutations,
            seed=args.seed + i,
        )
        summary.append(result)

    summary_path = args.out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Ablation Summary",
        "",
        f"- Permutations per ablation: `{args.n_permutations}`",
        "",
        "| Ablation | Status | Rows | Chapters | AUC | Perm p | BinSeg Break | Near 80 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        if item.get("status") != "ok":
            md_lines.append(
                f"| {item['name']} | {item['status']} | - | - | - | - | - | - |"
            )
            continue
        md_lines.append(
            f"| {item['name']} | ok | {item['rows']} | {item['chapters']} | "
            f"{item['auc_mean']:.4f} | {item['perm_p']:.4f} | {item['break_binseg']} | "
            f"{str(item['boundary_near_80'])} |"
        )

    md_path = args.out_dir / "ablation_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Ablation summary JSON: {summary_path}")
    print(f"Ablation summary markdown: {md_path}")


if __name__ == "__main__":
    main()
