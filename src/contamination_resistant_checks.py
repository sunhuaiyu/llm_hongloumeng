#!/usr/bin/env python3
"""Contamination-resistant checks for chapter boundary hypothesis."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def require_dependencies():
    try:
        import jieba  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing deps. Install with:\n"
            "source venv/bin/activate && pip install -U torch sentence-transformers jieba"
        ) from exc


def load_chapters(chapters_dir: Path):
    import pandas as pd

    rows = []
    for p in sorted(chapters_dir.glob("chapter_*.txt")):
        cid = int(p.stem.split("_")[-1])
        text = p.read_text(encoding="utf-8")
        rows.append(
            {
                "chapter_id": cid,
                "true_period_label": 1 if cid > 80 else 0,
                "text": text,
            }
        )
    if len(rows) != 120:
        raise SystemExit(f"Expected 120 chapter files, got {len(rows)} in {chapters_dir}")
    return pd.DataFrame(rows).sort_values("chapter_id").reset_index(drop=True)


def best_flip_accuracy(pred, true):
    import numpy as np

    pred = np.asarray(pred).astype(int)
    true = np.asarray(true).astype(int)
    acc0 = float((pred == true).mean())
    acc1 = float((1 - pred == true).mean())
    if acc1 > acc0:
        return acc1, (1 - pred)
    return acc0, pred


def best_split_for_order(labels_ordered):
    import numpy as np

    y = np.asarray(labels_ordered).astype(int)
    n = len(y)
    best_t = 1
    best_err = 10**9
    for t in range(1, n):
        split = (np.arange(n) >= t).astype(int)
        err0 = int((split != y).sum())
        err1 = int((split != (1 - y)).sum())
        err = min(err0, err1)
        if err < best_err:
            best_err = err
            best_t = t
    return int(best_t), int(best_err)


def eval_cluster_methods(X, true_labels, chapter_ids, seed=42):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y_true = np.asarray(true_labels).astype(int)
    chapter_ids = np.asarray(chapter_ids).astype(int)
    Xz = StandardScaler().fit_transform(X)

    methods = []
    labels_map = {}

    km = KMeans(n_clusters=2, n_init=20, random_state=seed).fit(Xz)
    labels_map["kmeans_2"] = km.labels_.astype(int)

    agg = AgglomerativeClustering(n_clusters=2).fit(Xz)
    labels_map["agglomerative_2"] = agg.labels_.astype(int)

    try:
        gmm = GaussianMixture(
            n_components=2,
            random_state=seed,
            covariance_type="diag",
            reg_covar=1e-4,
        ).fit(Xz.astype("float64"))
        labels_map["gmm_2"] = gmm.predict(Xz.astype("float64")).astype(int)
    except Exception:
        pass

    for name, labels in labels_map.items():
        if len(np.unique(labels)) < 2:
            sil = float("nan")
        else:
            sil = float(silhouette_score(Xz, labels))
        ari = float(adjusted_rand_score(y_true, labels))
        nmi = float(normalized_mutual_info_score(y_true, labels))
        acc, aligned = best_flip_accuracy(labels, y_true)
        split_t, split_err = best_split_for_order(aligned[np.argsort(chapter_ids)])
        transitions = int(np.sum(np.diff(aligned[np.argsort(chapter_ids)]) != 0))
        methods.append(
            {
                "method": name,
                "silhouette": sil,
                "ari": ari,
                "nmi": nmi,
                "best_flip_accuracy": float(acc),
                "best_split_chapter": int(split_t),
                "best_split_error_count": int(split_err),
                "label_transitions_across_chapters": transitions,
            }
        )
        labels_map[name] = aligned

    methods = sorted(methods, key=lambda x: x["best_flip_accuracy"], reverse=True)
    best_method = methods[0]["method"]
    return methods, labels_map, best_method, Xz


def run_null_test(labels_aligned, n_perm=5000, center=80, tol=5, seed=42):
    import numpy as np

    rng = np.random.default_rng(seed)
    y = np.asarray(labels_aligned).astype(int)
    obs_t, obs_err = best_split_for_order(y)

    near = 0
    better_err = 0
    near_and_better = 0
    split_counts = Counter()
    for _ in range(n_perm):
        yp = rng.permutation(y)
        t, err = best_split_for_order(yp)
        split_counts[t] += 1
        is_near = abs(t - center) <= tol
        if is_near:
            near += 1
        if err <= obs_err:
            better_err += 1
            if is_near:
                near_and_better += 1

    return {
        "observed_best_split_chapter": int(obs_t),
        "observed_best_split_error": int(obs_err),
        "observed_near_80": bool(abs(obs_t - center) <= tol),
        "null_near_80_rate": float(near / n_perm),
        "null_err_le_observed_rate": float(better_err / n_perm),
        "null_near_80_and_err_le_observed_rate": float(near_and_better / n_perm),
        "n_permutations": int(n_perm),
        "split_histogram": {str(k): int(v) for k, v in sorted(split_counts.items())},
    }


def mask_names(text, names):
    s = text
    for n in names:
        s = s.replace(n, "人名")
    return s


def is_cjk_token(tok: str) -> bool:
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,}", tok))


def preprocess_topic_suppressed(texts, top_k=400):
    import jieba

    tokenized = [jieba.lcut(t, cut_all=False) for t in texts]
    cnt = Counter()
    for toks in tokenized:
        for tok in toks:
            if is_cjk_token(tok):
                cnt[tok] += 1
    topic_terms = {tok for tok, _ in cnt.most_common(top_k)}

    processed = []
    for toks in tokenized:
        kept = []
        for tok in toks:
            if tok in topic_terms and len(tok) >= 2:
                continue
            kept.append(tok)
        out = " ".join(kept).strip()
        processed.append(out if out else "空")
    return processed, topic_terms


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    require_dependencies()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapters-dir", type=Path, default=Path("data/chapters"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/contamination_checks"))
    parser.add_argument("--e5-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--null-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topic-top-k", type=int, default=400)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chapters = load_chapters(args.chapters_dir)
    chapter_ids = chapters["chapter_id"].to_numpy()
    y_true = chapters["true_period_label"].to_numpy()

    # Check 1: corpus-only TF-IDF char-ngrams clustering.
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=30000)
    X_sparse = tfidf.fit_transform(chapters["text"].tolist())
    n_comp = max(10, min(100, X_sparse.shape[0] - 1, X_sparse.shape[1] - 1))
    X_tfidf = normalize(TruncatedSVD(n_components=n_comp, random_state=args.seed).fit_transform(X_sparse))
    tfidf_metrics, tfidf_labels_map, tfidf_best_method, _ = eval_cluster_methods(
        X_tfidf, y_true, chapter_ids, seed=args.seed
    )
    tfidf_null = run_null_test(
        labels_aligned=tfidf_labels_map[tfidf_best_method][np.argsort(chapter_ids)],
        n_perm=args.null_permutations,
        seed=args.seed,
    )

    # Check 2: name-masked + topic-suppressed embedding clustering.
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
    masked_texts = [mask_names(t, names) for t in chapters["text"].tolist()]
    suppressed_texts, topic_terms = preprocess_topic_suppressed(masked_texts, top_k=args.topic_top_k)

    print(f"Loading embedding model: {args.e5_model}")
    embedder = SentenceTransformer(args.e5_model, device="cpu")
    emb_input = [f"query: {t}" for t in suppressed_texts]
    X_emb = embedder.encode(
        emb_input,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    X_emb = np.asarray(X_emb, dtype=np.float32)
    emb_metrics, emb_labels_map, emb_best_method, _ = eval_cluster_methods(
        X_emb, y_true, chapter_ids, seed=args.seed
    )
    emb_null = run_null_test(
        labels_aligned=emb_labels_map[emb_best_method][np.argsort(chapter_ids)],
        n_perm=args.null_permutations,
        seed=args.seed + 1,
    )

    # Optional: baseline e5 unsupervised result null-test (if existing).
    baseline_e5 = None
    baseline_path = Path("artifacts/embedding_cluster_e5_small/chapter_cluster_assignments.parquet")
    baseline_metrics_path = Path("artifacts/embedding_cluster_e5_small/cluster_metrics.json")
    if baseline_path.exists() and baseline_metrics_path.exists():
        base_df = pd.read_parquet(baseline_path).sort_values("chapter_id").reset_index(drop=True)
        base_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        best_m = base_metrics.get("best_method", "gmm_2")
        col = f"cluster_{best_m}"
        if col in base_df.columns:
            baseline_e5 = {
                "best_method": best_m,
                "null_test": run_null_test(
                    labels_aligned=base_df[col].to_numpy(dtype=int),
                    n_perm=args.null_permutations,
                    seed=args.seed + 2,
                ),
            }

    summary = {
        "check_1_tfidf_unsupervised": {
            "metrics": tfidf_metrics,
            "best_method": tfidf_best_method,
            "null_test": tfidf_null,
            "n_features_tfidf": int(X_sparse.shape[1]),
            "svd_components": int(n_comp),
        },
        "check_2_masked_topic_suppressed_embedding": {
            "embedding_model": args.e5_model,
            "metrics": emb_metrics,
            "best_method": emb_best_method,
            "null_test": emb_null,
            "topic_terms_removed_count": int(len(topic_terms)),
            "topic_top_k": int(args.topic_top_k),
        },
        "check_3_baseline_e5_null_test": baseline_e5,
    }

    save_json(summary, args.out_dir / "summary.json")

    # Figure: null split histogram for masked/topic-suppressed embedding best method.
    hist = emb_null["split_histogram"]
    xs = np.array(sorted(int(k) for k in hist.keys()))
    ys = np.array([hist[str(x)] for x in xs], dtype=float)
    plt.figure(figsize=(10, 4))
    plt.bar(xs, ys, width=0.8)
    plt.axvline(80, color="red", linestyle="--", linewidth=1.5, label="Chapter 80")
    plt.axvspan(75, 85, color="red", alpha=0.12, label="Near-80 band")
    plt.xlabel("Best split chapter under shuffled order")
    plt.ylabel("Count")
    plt.title("Null Distribution of Best Split (Masked + Topic-Suppressed Embedding)")
    plt.legend()
    fig_path = args.out_dir / "null_split_hist_masked_embedding.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # Markdown report.
    def fmt_metric_block(metrics):
        lines = []
        for m in metrics:
            lines.append(
                f"| {m['method']} | {m['silhouette']:.4f} | {m['ari']:.4f} | {m['nmi']:.4f} | "
                f"{m['best_flip_accuracy']:.4f} | {m['best_split_chapter']} | {m['best_split_error_count']} | "
                f"{m['label_transitions_across_chapters']} |"
            )
        return lines

    md = []
    md.append("# Contamination-Resistant Checks")
    md.append("")
    md.append("## Check 1: Corpus-Only TF-IDF Char-Ngram Unsupervised Clustering")
    md.append("| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    md.extend(fmt_metric_block(tfidf_metrics))
    md.append(
        f"- Null near-80 rate: `{tfidf_null['null_near_80_rate']:.4f}` "
        f"(n={tfidf_null['n_permutations']})"
    )
    md.append("")
    md.append("## Check 2: Name-Masked + Topic-Suppressed Embedding Unsupervised Clustering")
    md.append(f"- Embedding model: `{args.e5_model}`")
    md.append(f"- Topic terms removed: `{len(topic_terms)}`")
    md.append("| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Transitions |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    md.extend(fmt_metric_block(emb_metrics))
    md.append(
        f"- Null near-80 rate: `{emb_null['null_near_80_rate']:.4f}` "
        f"(n={emb_null['n_permutations']})"
    )
    md.append(
        f"- Observed best split: `{emb_null['observed_best_split_chapter']}` "
        f"(near 80: `{emb_null['observed_near_80']}`)"
    )
    if baseline_e5 is not None:
        b = baseline_e5["null_test"]
        md.append("")
        md.append("## Check 3: Baseline E5 Label-Order Null Test")
        md.append(f"- Baseline best method: `{baseline_e5['best_method']}`")
        md.append(f"- Observed best split chapter: `{b['observed_best_split_chapter']}`")
        md.append(f"- Null near-80 rate: `{b['null_near_80_rate']:.4f}`")
        md.append(
            f"- Null near-80 and error<=observed rate: "
            f"`{b['null_near_80_and_err_le_observed_rate']:.4f}`"
        )
    md.append("")
    md.append("## Outputs")
    md.append(f"- Summary JSON: `{args.out_dir / 'summary.json'}`")
    md.append(f"- Null histogram figure: `{fig_path}`")
    report_path = args.out_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Saved: {args.out_dir / 'summary.json'}")
    print(f"Saved: {report_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
