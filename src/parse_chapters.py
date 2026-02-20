#!/usr/bin/env python3
"""Parse Hongloumeng plain text into chapter files and QC metadata."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean

CHAPTER_HEADER_RE = re.compile(
    r"^第([0-9一二三四五六七八九十百〇零两]+)章(?:\s+|$)(.*)$"
)
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?]")
DIALOGUE_RE = re.compile(r"[“\"「『](.*?)[”\"」』]")
WHITESPACE_RE = re.compile(r"\s+")


def sentence_lengths(text: str) -> list[int]:
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text)]
    return [len(p) for p in parts if p]


def dialogue_char_count(text: str) -> int:
    return sum(len(m.group(1)) for m in DIALOGUE_RE.finditer(text))


def parse_chapters(input_path: Path) -> list[dict]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    chapters: list[dict] = []

    current_header = None
    current_title = None
    current_start = None
    current_lines: list[str] = []

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        match = CHAPTER_HEADER_RE.match(stripped)
        if match:
            if current_header is not None:
                chapters.append(
                    {
                        "chapter_id": len(chapters) + 1,
                        "header": current_header,
                        "title": current_title,
                        "line_start": current_start,
                        "line_end": line_no - 1,
                        "text": "\n".join(current_lines).strip() + "\n",
                    }
                )

            current_header = stripped
            current_title = match.group(2).strip() or stripped
            current_start = line_no
            current_lines = [raw_line]
            continue

        if current_header is not None:
            current_lines.append(raw_line)

    if current_header is not None:
        chapters.append(
            {
                "chapter_id": len(chapters) + 1,
                "header": current_header,
                "title": current_title,
                "line_start": current_start,
                "line_end": len(lines),
                "text": "\n".join(current_lines).strip() + "\n",
            }
        )

    if not chapters:
        raise ValueError(f"No chapter headers matched in {input_path}")

    return chapters


def write_outputs(chapters: list[dict], out_dir: Path, metadata_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chapter_id",
                "period_label",
                "header",
                "title",
                "line_start",
                "line_end",
                "char_count_non_ws",
                "sentence_count",
                "mean_sentence_len",
                "dialogue_char_ratio",
            ],
        )
        writer.writeheader()

        for chapter in chapters:
            chapter_id = chapter["chapter_id"]
            period_label = 1 if chapter_id > 80 else 0
            text = chapter["text"]
            compact = WHITESPACE_RE.sub("", text)
            non_ws_count = len(compact)
            sent_lens = sentence_lengths(text)
            mean_sent_len = mean(sent_lens) if sent_lens else 0.0
            dialog_chars = dialogue_char_count(text)
            dialog_ratio = (dialog_chars / non_ws_count) if non_ws_count else 0.0

            chapter_file = out_dir / f"chapter_{chapter_id:03d}.txt"
            chapter_file.write_text(text, encoding="utf-8")

            writer.writerow(
                {
                    "chapter_id": chapter_id,
                    "period_label": period_label,
                    "header": chapter["header"],
                    "title": chapter["title"],
                    "line_start": chapter["line_start"],
                    "line_end": chapter["line_end"],
                    "char_count_non_ws": non_ws_count,
                    "sentence_count": len(sent_lens),
                    "mean_sentence_len": round(mean_sent_len, 4),
                    "dialogue_char_ratio": round(dialog_ratio, 6),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="HongLouMeng.txt",
        type=Path,
        help="Input novel text file.",
    )
    parser.add_argument(
        "--chapters-dir",
        default=Path("data/chapters"),
        type=Path,
        help="Output directory for chapter text files.",
    )
    parser.add_argument(
        "--metadata",
        default=Path("artifacts/reports/chapter_qc.csv"),
        type=Path,
        help="Output chapter metadata CSV.",
    )
    args = parser.parse_args()

    chapters = parse_chapters(args.input)
    write_outputs(chapters, args.chapters_dir, args.metadata)

    print(f"Parsed chapters: {len(chapters)}")
    print(f"Chapter files: {args.chapters_dir}")
    print(f"QC metadata: {args.metadata}")


if __name__ == "__main__":
    main()
