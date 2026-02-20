#!/usr/bin/env python3
"""Unsupervised clustering analysis in embedding space for Hongloumeng."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_dependencies() -> None:
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import seaborn  # noqa: F401
        import umap  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "source venv/bin/activate && pip install -U torch sentence-transformers umap-learn"
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
    raise FileNotFoundError(f"Missing input file: {path}")


def best_flip_accuracy(pred, true):
    import numpy as np

    pred = np.asarray(pred).astype(int)
    true = np.asarray(true).astype(int)
    acc0 = float((pred == true).mean())
    acc1 = float((1 - pred == true).mean())
    if acc1 > acc0:
        return acc1, (1 - pred)
    return acc0, pred


def best_boundary_from_labels(chapter_ids, pred_labels):
    import numpy as np

    order = np.argsort(chapter_ids)
    ch = np.asarray(chapter_ids)[order]
    y = np.asarray(pred_labels)[order]
    best_err = 10**9
    best_t = None
    for t in range(1, 120):
        split = (ch > t).astype(int)
        err0 = int((split != y).sum())
        err1 = int((split != (1 - y)).sum())
        err = min(err0, err1)
        if err < best_err:
            best_err = err
            best_t = t
    return int(best_t), int(best_err)


def eval_clustering(name, X, true_labels, chapter_ids, pred_labels):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    X = np.asarray(X)
    pred_labels = np.asarray(pred_labels).astype(int)
    true_labels = np.asarray(true_labels).astype(int)

    if len(np.unique(pred_labels)) < 2:
        sil = float("nan")
    else:
        sil = float(silhouette_score(X, pred_labels))

    ari = float(adjusted_rand_score(true_labels, pred_labels))
    nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
    acc, aligned = best_flip_accuracy(pred_labels, true_labels)
    split_t, split_err = best_boundary_from_labels(chapter_ids, aligned)
    transitions = int(np.sum(np.diff(aligned[np.argsort(chapter_ids)]) != 0))

    return {
        "method": name,
        "silhouette": sil,
        "ari": ari,
        "nmi": nmi,
        "best_flip_accuracy": float(acc),
        "best_split_chapter": int(split_t),
        "best_split_error_count": int(split_err),
        "label_transitions_across_chapters": transitions,
    }, aligned


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


def main() -> None:
    require_dependencies()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("artifacts/features/stylometry_chunk.parquet"),
        help="Chunk table with chunk_text/chapter_id/period_label.",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="Sentence-transformers embedding model id.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/embedding_cluster"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunk_df = load_table(args.chunks)
    need_cols = {"chapter_id", "period_label", "chunk_id", "chunk_text"}
    miss = need_cols - set(chunk_df.columns)
    if miss:
        raise SystemExit(f"Missing required columns in chunk table: {sorted(miss)}")

    chunk_df = chunk_df[["chapter_id", "period_label", "chunk_id", "chunk_text"]].copy()
    chunk_df["chunk_text"] = chunk_df["chunk_text"].astype(str).str.slice(0, args.max_text_chars)
    chunk_df = chunk_df.sort_values(["chapter_id", "chunk_id"]).reset_index(drop=True)

    print(f"Loading embedding model: {args.model}")
    embedder = SentenceTransformer(args.model, device=args.device)
    chunk_texts = chunk_df["chunk_text"].tolist()
    print(f"Encoding chunk texts: {len(chunk_texts)}")
    chunk_emb = embedder.encode(
        chunk_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    chunk_emb = np.asarray(chunk_emb, dtype=np.float32)

    emb_cols = [f"emb_{i:04d}" for i in range(chunk_emb.shape[1])]
    chunk_emb_df = pd.concat(
        [
            chunk_df.reset_index(drop=True),
            pd.DataFrame(chunk_emb, columns=emb_cols),
        ],
        axis=1,
    )

    chapter_emb_df = (
        chunk_emb_df.groupby("chapter_id", as_index=False)[emb_cols]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    chapter_labels = (
        chunk_df.groupby("chapter_id", as_index=False)["period_label"]
        .first()
        .rename(columns={"period_label": "true_period_label"})
    )
    chapter_emb_df = chapter_emb_df.merge(chapter_labels, on="chapter_id", how="left")

    X = chapter_emb_df[emb_cols].to_numpy(dtype=np.float32)
    y_true = chapter_emb_df["true_period_label"].to_numpy(dtype=int)
    chapter_ids = chapter_emb_df["chapter_id"].to_numpy(dtype=int)

    Xz = StandardScaler().fit_transform(X)
    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=args.seed,
    ).fit_transform(Xz)

    km = KMeans(n_clusters=2, n_init=20, random_state=args.seed).fit(Xz)
    agg = AgglomerativeClustering(n_clusters=2).fit(Xz)
    results = []
    aligned_labels = {}
    method_labels = [
        ("kmeans_2", km.labels_),
        ("agglomerative_2", agg.labels_),
    ]
    try:
        gmm = GaussianMixture(
            n_components=2,
            random_state=args.seed,
            covariance_type="diag",
            reg_covar=1e-4,
        ).fit(Xz.astype(np.float64))
        method_labels.append(("gmm_2", gmm.predict(Xz.astype(np.float64))))
    except Exception as exc:
        print(f"Skipping gmm_2 due to fit failure: {exc}")

    for name, labels in method_labels:
        metrics, aligned = eval_clustering(name, Xz, y_true, chapter_ids, labels)
        results.append(metrics)
        aligned_labels[name] = aligned

    result_df = pd.DataFrame(results).sort_values("best_flip_accuracy", ascending=False)
    best_method = str(result_df.iloc[0]["method"])
    best_labels = aligned_labels[best_method]

    chapter_out = chapter_emb_df[["chapter_id", "true_period_label"]].copy()
    chapter_out["umap_x"] = umap_2d[:, 0]
    chapter_out["umap_y"] = umap_2d[:, 1]
    for name, labels in aligned_labels.items():
        chapter_out[f"cluster_{name}"] = labels.astype(int)

    chunk_emb_path = save_table(chunk_emb_df, args.out_dir / "chunk_embeddings")
    chapter_emb_path = save_table(chapter_emb_df, args.out_dir / "chapter_embeddings")
    chapter_cluster_path = save_table(chapter_out, args.out_dir / "chapter_cluster_assignments")
    result_path = args.out_dir / "cluster_metrics.json"
    result_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "n_chunks": int(len(chunk_df)),
                "n_chapters": int(len(chapter_emb_df)),
                "metrics": results,
                "best_method": best_method,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    sns.set_theme(style="whitegrid")
    # Figure 1: UMAP by best cluster
    plt.figure(figsize=(8, 6))
    palette = {0: "#1f77b4", 1: "#d62728"}
    sns.scatterplot(
        x=chapter_out["umap_x"],
        y=chapter_out["umap_y"],
        hue=chapter_out[f"cluster_{best_method}"],
        palette=palette,
        s=45,
    )
    plt.title(f"Chapter Embeddings UMAP Colored by {best_method}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Cluster")
    fig1 = args.out_dir / "umap_best_cluster.png"
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()

    # Figure 2: UMAP by true 1-80 vs 81-120 split.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=chapter_out["umap_x"],
        y=chapter_out["umap_y"],
        hue=chapter_out["true_period_label"],
        palette=palette,
        s=45,
    )
    plt.title("Chapter Embeddings UMAP Colored by True Period Label")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="True Label")
    fig2 = args.out_dir / "umap_true_period.png"
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()

    # Figure 3: chapter timeline for best cluster.
    order = np.argsort(chapter_out["chapter_id"].to_numpy())
    t_ch = chapter_out["chapter_id"].to_numpy()[order]
    t_lb = chapter_out[f"cluster_{best_method}"].to_numpy()[order]
    plt.figure(figsize=(10, 2.4))
    plt.scatter(t_ch, t_lb, c=[palette[int(v)] for v in t_lb], s=28)
    plt.axvline(x=80, color="black", linestyle="--", linewidth=1.2, label="Chapter 80")
    plt.yticks([0, 1], ["Cluster 0", "Cluster 1"])
    plt.xlabel("Chapter")
    plt.title(f"Chapter Order vs Cluster Assignment ({best_method})")
    plt.legend(loc="upper right")
    fig3 = args.out_dir / "chapter_timeline_best_cluster.png"
    plt.tight_layout()
    plt.savefig(fig3, dpi=160)
    plt.close()

    md_lines = [
        "# Embedding Clustering Report",
        "",
        f"- Embedding model: `{args.model}`",
        f"- Chunks embedded: `{len(chunk_df)}`",
        f"- Chapter embeddings: `{len(chapter_emb_df)}`",
        f"- Best method (by best_flip_accuracy): `{best_method}`",
        "",
        "| Method | Silhouette | ARI | NMI | Best-Flip Accuracy | Best Split Chapter | Split Errors | Label Transitions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in result_df.iterrows():
        md_lines.append(
            f"| {r['method']} | {r['silhouette']:.4f} | {r['ari']:.4f} | {r['nmi']:.4f} | "
            f"{r['best_flip_accuracy']:.4f} | {int(r['best_split_chapter'])} | "
            f"{int(r['best_split_error_count'])} | {int(r['label_transitions_across_chapters'])} |"
        )
    md_lines.extend(
        [
            "",
            "## Outputs",
            f"- Chunk embeddings: `{chunk_emb_path}`",
            f"- Chapter embeddings: `{chapter_emb_path}`",
            f"- Chapter clusters: `{chapter_cluster_path}`",
            f"- Metrics JSON: `{result_path}`",
            f"- Figure: `{fig1}`",
            f"- Figure: `{fig2}`",
            f"- Figure: `{fig3}`",
        ]
    )
    report_path = args.out_dir / "report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {chunk_emb_path}")
    print(f"Saved: {chapter_emb_path}")
    print(f"Saved: {chapter_cluster_path}")
    print(f"Saved: {result_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {fig1}")
    print(f"Saved: {fig2}")
    print(f"Saved: {fig3}")


if __name__ == "__main__":
    main()
