import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi
from ranx import Qrels, Run, evaluate


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def load_chunks(chunk_dir: Path) -> List[Dict]:
    docs = []
    for fp in sorted(chunk_dir.glob("*.json")):
        arr = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(arr, list):
            docs.extend(arr)
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth JSONL")
    parser.add_argument("--chunk_dir", required=True, help="Chunk directory for one strategy")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", required=True, help="Output metrics JSON")
    args = parser.parse_args()

    gt_rows = read_jsonl(Path(args.gt))
    chunks = load_chunks(Path(args.chunk_dir))
    if not chunks:
        raise RuntimeError(f"No chunks found in: {args.chunk_dir}")

    doc_ids = [c.get("chunk_id", f"doc_{i}") for i, c in enumerate(chunks)]
    corpus = [tokenize(c.get("content", "")) for c in chunks]
    bm25 = BM25Okapi(corpus)

    qrels_dict = {}
    run_dict = {}

    used = 0
    for row in gt_rows:
        qid = row["id"]
        q = row.get("question", "")
        rel_ids = row.get("gold_chunk_ids", [])  # preferred for chunk eval
        if not rel_ids:
            continue

        scores = bm25.get_scores(tokenize(q))
        rank_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(args.k, 50)]

        qrels_dict[qid] = {rid: 1 for rid in rel_ids}
        run_dict[qid] = {doc_ids[i]: float(scores[i]) for i in rank_idx}
        used += 1

    if used == 0:
        raise RuntimeError("No usable questions. Please provide gold_chunk_ids in ground truth.")

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    metrics = evaluate(
        qrels,
        run,
        metrics=[f"recall@{args.k}", f"precision@{args.k}", f"ndcg@{args.k}", f"mrr@{args.k}", f"map@{args.k}"],
    )

    out = {
        "num_questions_used": used,
        "k": args.k,
        "metrics": metrics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()