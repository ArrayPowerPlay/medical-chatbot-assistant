import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.dataset_builder.parent_child_chunker import AdaptiveChunker


def iter_corpus(path: Path, limit: int | None = None) -> Iterator[dict]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and count >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            pmid = str(record.get("pmid", "")).strip()
            title = str(record.get("title", "")).strip()
            abstract = str(record.get("abstractText", "")).strip()

            if not pmid:
                continue

            yield {
                "pmid": pmid,
                "title": title,
                "abstractText": abstract,
            }
            count += 1


def chunk_corpus(
    input_path: Path,
    output_path: Path,
    mode: str,
    limit: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for article in iter_corpus(input_path, limit=limit):
            result = AdaptiveChunker.process_article(
                article_id=article["pmid"],
                title=article["title"],
                abstract=article["abstractText"],
            )

            if mode in {"parent", "both"}:
                for parent in result["parents"]:
                    out.write(json.dumps({
                        "type": "parent",
                        "pmid": parent["pmid"],
                        "parent_id": parent["parent_id"],
                        "title": parent["title"],
                        "text": parent["text"],
                    }, ensure_ascii=False) + "\n")

            if mode in {"child", "both"}:
                for parent_id, child_texts in result["children"].items():
                    for idx, child_text in enumerate(child_texts):
                        out.write(json.dumps({
                            "type": "child",
                            "pmid": article["pmid"],
                            "parent_id": parent_id,
                            "child_index": idx,
                            "title": article["title"],
                            "text": child_text,
                        }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk corpus.jsonl into parent chunks, child chunks, or both."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/corpus/corpus.jsonl"),
        help="Path to input corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunked_corpus.jsonl"),
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--mode",
        choices=["parent", "child", "both"],
        default="both",
        help="Choose which chunk type to export",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of articles to process",
    )

    args = parser.parse_args()

    chunk_corpus(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        limit=args.limit,
    )

    print(f"Saved chunked data to {args.output}")


if __name__ == "__main__":
    main()