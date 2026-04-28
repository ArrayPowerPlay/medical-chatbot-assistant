#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.dataset_builder.parent_child_chunker import AdaptiveChunker

def find_article_by_pmid(corpus_path: Path, pmid: str):
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(doc.get("pmid", "")).strip() == pmid:
                return {
                    "pmid": pmid,
                    "title": doc.get("title", ""),
                    "abstractText": doc.get("abstractText", "")
                }
    return None

def dump_result(result: dict, out_path: Path | None, mode: str):
    lines = []
    if mode in ("parent", "both"):
        for p in result["parents"]:
            lines.append(json.dumps({
                "type": "parent",
                "pmid": p["pmid"],
                "parent_id": p["parent_id"],
                "title": p["title"],
                "text": p["text"],
            }, ensure_ascii=False))
    if mode in ("child", "both"):
        for parent_id, childs in result["children"].items():
            for idx, c in enumerate(childs):
                lines.append(json.dumps({
                    "type": "child",
                    "pmid": result["parents"][0]["pmid"] if result["parents"] else "",
                    "parent_id": parent_id,
                    "child_index": idx,
                    "text": c,
                }, ensure_ascii=False))

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as of:
            of.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"Wrote {len(lines)} chunk(s) to {out_path}")
    else:
        for l in lines:
            print(l)

def main():
    p = argparse.ArgumentParser(description="Chunk single article from corpus.jsonl by pmid")
    p.add_argument("pmid", help="pmid of the article to chunk")
    p.add_argument("--input", type=Path, default=Path("data/corpus/corpus.jsonl"))
    p.add_argument("--mode", choices=["parent", "child", "both"], default="both")
    p.add_argument("--output", type=Path, default=None, help="optional output JSONL path")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input corpus not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    article = find_article_by_pmid(args.input, args.pmid)
    if not article:
        print(f"Article with pmid={args.pmid} not found in {args.input}", file=sys.stderr)
        sys.exit(1)

    result = AdaptiveChunker.process_article(article_id=article["pmid"], title=article["title"], abstract=article["abstractText"])
    dump_result(result, args.output, args.mode)

if __name__ == "__main__":
    main()