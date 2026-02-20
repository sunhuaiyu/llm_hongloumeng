#!/usr/bin/env python3
"""Run change-point and classification tests for the authorship hypothesis."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def require_dependencies():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import ruptures as rpt  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: F401
        from sklearn.model_selection import GroupKFold  # noqa: F401
        from sklearn.pipeline import Pipeline  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "source venv/bin/activate && "
            "pip install -U pandas numpy scikit-learn scipy ruptures matplotlib pyarrow"
        ) from exc


def load_table(path: Path):
    import pandas as pd

    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)

    if path.suffix == ".parquet":
        alt = path.with_suffix(".csv")
    else:
        alt = path.with_suffix(".parquet")
    if alt.exists():
        if alt.suffix == ".parquet":
            return pd.read_parquet(alt)
        return pd.read_csv(alt)

    raise FileNotFoundError(f"Missing input table: {path}")


@dataclass
class EvalResult:
    auc_mean: float
    auc_std: float
    bacc_mean: float
    bacc_std: float
    folds_used: int


def evaluate_grouped_classifier(X, y, groups, n_splits: int = 5):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return EvalResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    if len(np.unique(y)) < 2:
        return EvalResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    n_splits = max(2, min(n_splits, len(unique_groups)))

    aucs = []
    baccs = []

    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, solver="lbfgs")),
            ]
        )
        try:
            model.fit(X_train, y_train)
        except ValueError:
            continue
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        if len(set(y_test.tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y_test, proba))
        baccs.append(balanced_accuracy_score(y_test, pred))

    if not aucs:
        return EvalResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    return EvalResult(
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        bacc_mean=float(np.mean(baccs)),
        bacc_std=float(np.std(baccs)),
        folds_used=len(aucs),
    )


def chapter_permutation_pvalue(
    X,
    y,
    groups,
    observed_auc: float,
    n_perm: int,
    seed: int,
    label: str = "perm",
    progress_every: int = 0,
):
    import numpy as np

    rng = np.random.default_rng(seed)
    chapter_ids = np.array(sorted(set(groups.tolist())))
    chapter_to_label = {cid: int(y[groups == cid][0]) for cid in chapter_ids}
    original_labels = np.array([chapter_to_label[cid] for cid in chapter_ids], dtype=int)

    null_aucs = []
    for i in range(n_perm):
        permuted = rng.permutation(original_labels)
        label_map = {cid: int(lbl) for cid, lbl in zip(chapter_ids, permuted)}
        y_perm = np.array([label_map[cid] for cid in groups], dtype=int)
        eval_result = evaluate_grouped_classifier(X, y_perm, groups)
        null_aucs.append(eval_result.auc_mean)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[{label}] permutations completed: {i + 1}/{n_perm}")

    null_aucs = np.array(null_aucs, dtype=float)
    ge = np.sum(null_aucs >= observed_auc)
    p_value = (ge + 1) / (len(null_aucs) + 1)
    return float(p_value), null_aucs


def chapter_change_points(chapter_df, feature_cols):
    import numpy as np
    import ruptures as rpt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ordered = chapter_df.sort_values("chapter_id").reset_index(drop=True)
    X = ordered[feature_cols].to_numpy(dtype=float)

    X = StandardScaler().fit_transform(X)
    if X.shape[1] > 1:
        signal = PCA(n_components=1, random_state=0).fit_transform(X).ravel()
    else:
        signal = X.ravel()

    signal_2d = signal.reshape(-1, 1)
    binseg = rpt.Binseg(model="l2").fit(signal_2d)
    break_1 = int(binseg.predict(n_bkps=1)[0])

    pelt = rpt.Pelt(model="rbf").fit(signal_2d)
    penalty = 3.0 * math.log(max(len(signal), 2))
    pelt_breaks = [int(v) for v in pelt.predict(pen=penalty) if int(v) < len(signal)]

    return ordered["chapter_id"].to_numpy(), signal, break_1, pelt_breaks


def save_markdown(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    require_dependencies()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stylometry-chunk",
        type=Path,
        default=Path("artifacts/features/stylometry_chunk.parquet"),
    )
    parser.add_argument(
        "--stylometry-chapter",
        type=Path,
        default=Path("artifacts/features/stylometry_chapter.parquet"),
    )
    parser.add_argument(
        "--llm-chunk",
        type=Path,
        default=Path("artifacts/features/llm_signals_chunk.parquet"),
    )
    parser.add_argument("--out-reports", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--out-figures", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    style_chunk = load_table(args.stylometry_chunk)
    style_chapter = load_table(args.stylometry_chapter)

    style_cols = [c for c in style_chunk.columns if c.startswith("base_") or c.startswith("hgram_")]
    if not style_cols:
        raise SystemExit("No stylometry feature columns found.")

    X_style = style_chunk[style_cols].to_numpy(dtype=float)
    y_style = style_chunk["period_label"].to_numpy(dtype=int)
    g_style = style_chunk["chapter_id"].to_numpy(dtype=int)

    style_eval = evaluate_grouped_classifier(X_style, y_style, g_style)
    style_p, style_null = chapter_permutation_pvalue(
        X_style,
        y_style,
        g_style,
        style_eval.auc_mean,
        args.n_permutations,
        args.seed,
        label="stylometry",
        progress_every=10 if args.n_permutations >= 20 else 0,
    )

    ch_ids, signal, break_1, pelt_breaks = chapter_change_points(style_chapter, style_cols)

    llm_chunk = None
    if args.llm_chunk.exists() or args.llm_chunk.with_suffix(".csv").exists():
        try:
            llm_chunk = load_table(args.llm_chunk)
        except FileNotFoundError:
            llm_chunk = None

    llm_eval = None
    llm_p = None
    fused_eval = None
    fused_p = None

    if llm_chunk is not None:
        llm_cols = [c for c in llm_chunk.columns if c.startswith("llm_") and c.endswith("_mean")]
        if llm_cols:
            merged = style_chunk.merge(
                llm_chunk[["chapter_id", "chunk_id"] + llm_cols],
                on=["chapter_id", "chunk_id"],
                how="inner",
            ).dropna(subset=llm_cols)
            if not merged.empty:
                X_llm = merged[llm_cols].to_numpy(dtype=float)
                y_llm = merged["period_label"].to_numpy(dtype=int)
                g_llm = merged["chapter_id"].to_numpy(dtype=int)
                if len(np.unique(y_llm)) >= 2 and len(np.unique(g_llm)) >= 2:
                    llm_eval = evaluate_grouped_classifier(X_llm, y_llm, g_llm)
                    if not math.isnan(llm_eval.auc_mean):
                        llm_p, _ = chapter_permutation_pvalue(
                            X_llm,
                            y_llm,
                            g_llm,
                            llm_eval.auc_mean,
                            args.n_permutations,
                            args.seed + 1,
                            label="llm",
                            progress_every=10 if args.n_permutations >= 20 else 0,
                        )

                    fused_cols = style_cols + llm_cols
                    X_fused = merged[fused_cols].to_numpy(dtype=float)
                    fused_eval = evaluate_grouped_classifier(X_fused, y_llm, g_llm)
                    if not math.isnan(fused_eval.auc_mean):
                        fused_p, _ = chapter_permutation_pvalue(
                            X_fused,
                            y_llm,
                            g_llm,
                            fused_eval.auc_mean,
                            args.n_permutations,
                            args.seed + 2,
                            label="fused",
                            progress_every=10 if args.n_permutations >= 20 else 0,
                        )

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_reports.mkdir(parents=True, exist_ok=True)

    # Figure 1: chapter trajectory and breakpoints.
    plt.figure(figsize=(12, 4))
    plt.plot(ch_ids, signal, label="Chapter style trajectory (PC1)")
    plt.axvline(x=80, color="tab:red", linestyle="--", label="Hypothesis boundary (80)")
    plt.axvline(x=break_1, color="tab:green", linestyle="-.", label=f"BinSeg break ({break_1})")
    for bp in pelt_breaks:
        plt.axvline(x=bp, color="tab:orange", alpha=0.3)
    plt.xlabel("Chapter")
    plt.ylabel("Standardized signal")
    plt.title("Chapter-Level Style Trajectory and Change Points")
    plt.legend()
    fig_cp = args.out_figures / "chapter_change_points.png"
    plt.tight_layout()
    plt.savefig(fig_cp, dpi=150)
    plt.close()

    # Figure 2: permutation null.
    plt.figure(figsize=(8, 4))
    plt.hist(style_null, bins=30, alpha=0.8)
    plt.axvline(x=style_eval.auc_mean, color="tab:red", linestyle="--", label="Observed AUC")
    plt.xlabel("Null AUC")
    plt.ylabel("Frequency")
    plt.title("Stylometry AUC Permutation Null Distribution")
    plt.legend()
    fig_perm = args.out_figures / "stylometry_permutation_auc.png"
    plt.tight_layout()
    plt.savefig(fig_perm, dpi=150)
    plt.close()

    boundary_near_80 = abs(break_1 - 80) <= 5
    style_significant = (not math.isnan(style_p)) and style_p < 0.05 and style_eval.auc_mean >= 0.60
    llm_significant = (
        llm_eval is not None
        and llm_p is not None
        and (not math.isnan(llm_p))
        and llm_p < 0.05
        and llm_eval.auc_mean >= 0.60
    )

    if boundary_near_80 and style_significant and (llm_eval is None or llm_significant):
        decision = "support_H1"
    elif style_significant or llm_significant:
        decision = "partial_support"
    else:
        decision = "inconclusive"

    results = {
        "decision": decision,
        "boundary_near_80": boundary_near_80,
        "change_point_binseg": break_1,
        "change_points_pelt": pelt_breaks,
        "stylometry_eval": asdict(style_eval),
        "stylometry_permutation_p": style_p,
        "llm_eval": asdict(llm_eval) if llm_eval else None,
        "llm_permutation_p": llm_p,
        "fused_eval": asdict(fused_eval) if fused_eval else None,
        "fused_permutation_p": fused_p,
        "figures": {
            "change_points": str(fig_cp),
            "stylometry_permutation": str(fig_perm),
        },
    }

    json_path = args.out_reports / "results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Authorship Test Results",
        "",
        f"- Decision: `{decision}`",
        f"- BinSeg change-point chapter: `{break_1}`",
        f"- PELT change-points: `{pelt_breaks}`",
        f"- Boundary near 80 (+/-5): `{boundary_near_80}`",
        "",
        "## Stylometry",
        f"- AUC mean ± std: `{style_eval.auc_mean:.4f} ± {style_eval.auc_std:.4f}`",
        f"- Balanced accuracy mean ± std: `{style_eval.bacc_mean:.4f} ± {style_eval.bacc_std:.4f}`",
        f"- Permutation p-value: `{style_p:.6f}`",
        "",
    ]
    if llm_eval:
        lines.extend(
            [
                "## LLM Literary Signals",
                f"- AUC mean ± std: `{llm_eval.auc_mean:.4f} ± {llm_eval.auc_std:.4f}`",
                f"- Balanced accuracy mean ± std: `{llm_eval.bacc_mean:.4f} ± {llm_eval.bacc_std:.4f}`",
                f"- Permutation p-value: `{llm_p:.6f}`",
                "",
            ]
        )
    if fused_eval:
        lines.extend(
            [
                "## Fused Features (Stylometry + LLM)",
                f"- AUC mean ± std: `{fused_eval.auc_mean:.4f} ± {fused_eval.auc_std:.4f}`",
                f"- Balanced accuracy mean ± std: `{fused_eval.bacc_mean:.4f} ± {fused_eval.bacc_std:.4f}`",
                f"- Permutation p-value: `{fused_p:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            f"- JSON: `{json_path}`",
            f"- Figure: `{fig_cp}`",
            f"- Figure: `{fig_perm}`",
        ]
    )
    save_markdown(args.out_reports / "main_results.md", "\n".join(lines))

    print(f"Results JSON: {json_path}")
    print(f"Report: {args.out_reports / 'main_results.md'}")
    print(f"Figures: {fig_cp}, {fig_perm}")


if __name__ == "__main__":
    main()
