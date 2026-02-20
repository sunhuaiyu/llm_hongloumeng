#!/usr/bin/env python3
"""Build chunk-level and chapter-level stylometric features."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path


def require_dependencies():
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        from sklearn.feature_extraction.text import HashingVectorizer  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "source venv/bin/activate && "
            "pip install -U pandas numpy scikit-learn pyarrow"
        ) from exc


FUNCTION_CHARS = [
    "的",
    "了",
    "在",
    "是",
    "也",
    "又",
    "都",
    "而",
    "并",
    "且",
    "却",
    "便",
    "将",
    "于",
    "与",
    "其",
    "这",
    "那",
    "一个",
    "不",
    "无",
    "有",
    "人",
    "我",
    "你",
    "他",
]
PUNCT_CHARS = ["，", "。", "？", "！", "；", "：", "、", "…"]
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?]")
DIALOGUE_RE = re.compile(r"[“\"「『](.*?)[”\"」』]")
WS_RE = re.compile(r"\s+")


def sentence_lengths(text: str) -> list[int]:
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text)]
    return [len(p) for p in parts if p]


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def chunk_text(text: str, size: int, stride: int, min_chars: int) -> list[dict]:
    compact = text.strip()
    if len(compact) < min_chars:
        return []

    starts = list(range(0, max(len(compact) - size + 1, 1), stride))
    if not starts:
        starts = [0]

    last_start = max(len(compact) - size, 0)
    starts.append(last_start)

    chunks = []
    for idx, start in enumerate(sorted(set(starts)), start=1):
        end = min(start + size, len(compact))
        chunk = compact[start:end]
        if len(chunk) < min_chars and len(compact) > size:
            continue
        chunks.append(
            {
                "chunk_idx": idx,
                "start_char": start,
                "end_char": end,
                "chunk_text": chunk,
            }
        )

    if not chunks:
        chunks.append(
            {"chunk_idx": 1, "start_char": 0, "end_char": len(compact), "chunk_text": compact}
        )

    return chunks


def build_base_features(text: str) -> dict:
    compact = WS_RE.sub("", text)
    char_len = len(compact)
    sent_lens = sentence_lengths(compact)
    sent_count = len(sent_lens)
    sent_mean = (sum(sent_lens) / sent_count) if sent_count else 0.0
    sent_var = (
        sum((x - sent_mean) ** 2 for x in sent_lens) / sent_count if sent_count else 0.0
    )
    sent_std = math.sqrt(sent_var)

    counts = Counter(compact)
    unique_chars = len(counts)
    hapax = sum(1 for v in counts.values() if v == 1)
    dialogue_chars = sum(len(m.group(1)) for m in DIALOGUE_RE.finditer(text))
    quote_marks = sum(text.count(q) for q in ["“", "”", "\"", "「", "」", "『", "』"])

    feats = {
        "base_char_len": float(char_len),
        "base_sentence_count": float(sent_count),
        "base_sentence_mean_len": float(sent_mean),
        "base_sentence_std_len": float(sent_std),
        "base_unique_char_ratio": (unique_chars / char_len) if char_len else 0.0,
        "base_hapax_ratio": (hapax / unique_chars) if unique_chars else 0.0,
        "base_char_entropy": char_entropy(compact),
        "base_dialogue_char_ratio": (dialogue_chars / char_len) if char_len else 0.0,
        "base_quote_mark_density": (quote_marks / char_len) if char_len else 0.0,
    }

    for punct in PUNCT_CHARS:
        key = f"base_punct_{ord(punct)}_density"
        feats[key] = (compact.count(punct) / char_len) if char_len else 0.0

    for token in FUNCTION_CHARS:
        key = f"base_func_{token}_rate"
        feats[key] = (compact.count(token) / char_len) if char_len else 0.0

    return feats


def save_table(df, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    require_dependencies()

    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import HashingVectorizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapters-dir", type=Path, default=Path("data/chapters"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--chunk-stride", type=int, default=700)
    parser.add_argument("--min-chunk-chars", type=int, default=450)
    parser.add_argument("--ngram-features", type=int, default=512)
    args = parser.parse_args()

    chapter_files = sorted(args.chapters_dir.glob("chapter_*.txt"))
    if not chapter_files:
        raise SystemExit(f"No chapter files found in {args.chapters_dir}")

    rows = []
    for chapter_file in chapter_files:
        chapter_id = int(chapter_file.stem.split("_")[-1])
        text = chapter_file.read_text(encoding="utf-8")
        chunks = chunk_text(
            text=text,
            size=args.chunk_size,
            stride=args.chunk_stride,
            min_chars=args.min_chunk_chars,
        )
        for item in chunks:
            chunk_id = f"{chapter_id:03d}_{item['chunk_idx']:03d}"
            features = build_base_features(item["chunk_text"])
            row = {
                "chapter_id": chapter_id,
                "chunk_id": chunk_id,
                "period_label": 1 if chapter_id > 80 else 0,
                "start_char": item["start_char"],
                "end_char": item["end_char"],
                "chunk_text": item["chunk_text"],
            }
            row.update(features)
            rows.append(row)

    if not rows:
        raise SystemExit("No chunks were created. Adjust chunk parameters.")

    chunk_df = pd.DataFrame(rows)

    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        n_features=args.ngram_features,
        alternate_sign=False,
        norm=None,
    )
    mat = vectorizer.transform(chunk_df["chunk_text"].tolist()).astype(np.float32).toarray()
    lens = np.maximum(chunk_df["base_char_len"].to_numpy(dtype=np.float32), 1.0).reshape(-1, 1)
    mat = mat / lens

    ngram_cols = [f"hgram_{i:04d}" for i in range(args.ngram_features)]
    ngram_df = pd.DataFrame(mat, columns=ngram_cols)
    chunk_df = pd.concat([chunk_df.reset_index(drop=True), ngram_df], axis=1)

    numeric_cols = [
        c for c in chunk_df.columns if c.startswith("base_") or c.startswith("hgram_")
    ]
    chapter_df = (
        chunk_df.groupby("chapter_id", as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    chunk_counts = chunk_df.groupby("chapter_id").size().rename("chunk_count").reset_index()
    chapter_df = chapter_df.merge(chunk_counts, on="chapter_id", how="left")
    chapter_df["period_label"] = (chapter_df["chapter_id"] > 80).astype(int)

    out_chunk = save_table(chunk_df, args.out_dir / "stylometry_chunk")
    out_chapter = save_table(chapter_df, args.out_dir / "stylometry_chapter")

    print(f"Chunks: {len(chunk_df)}")
    print(f"Chapter aggregates: {len(chapter_df)}")
    print(f"Chunk features saved: {out_chunk}")
    print(f"Chapter features saved: {out_chapter}")


if __name__ == "__main__":
    main()
