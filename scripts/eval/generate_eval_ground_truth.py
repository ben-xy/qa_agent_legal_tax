import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                print(f"[WARN] Skipping invalid JSONL line {i}: {path}")
    return rows

def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # Support common wrappers like {"data":[...]} or {"items":[...]}
        for k in ("data", "items", "rows"):
            if isinstance(data.get(k), list):
                return [x for x in data[k] if isinstance(x, dict)]
    return []

def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pick(d: dict[str, Any], keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def to_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Try parsing JSON-encoded lists/objects
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                y = json.loads(s)
                return y if isinstance(y, list) else [y]
            except Exception:
                return [s]
        return [s]
    return [x]

def norm_row(r: dict[str, Any], idx: int) -> dict[str, Any] | None:
    question = pick(r, ["question", "query", "q", "user_query", "prompt"])
    gold_answer = pick(r, ["gold_answer", "answer", "reference_answer", "ground_truth", "gt_answer"])
    if not question or not gold_answer:
        return None

    rid = pick(r, ["id", "qid", "question_id", "uuid"])
    if not rid:
        h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
        rid = f"q_{idx}_{h}"

    refs = pick(r, ["references", "reference_chunks", "gold_chunks", "evidence", "citations"], [])
    refs = to_list(refs)

    out = {
        "id": str(rid),
        "qid": str(rid),
        "question": str(question),
        "gold_answer": str(gold_answer),
        "answer": str(gold_answer),  # Backward compatibility for scripts using "answer"
        "references": refs,
    }

    # Preserve optional retrieval supervision fields if present
    for k in ("gold_doc_ids", "gold_doc_id", "doc_ids", "source_ids", "meta"):
        if k in r and r[k] not in (None, ""):
            out[k] = r[k]
    return out

def load_rows(in_path: Path) -> list[dict[str, Any]]:
    suf = in_path.suffix.lower()
    if suf == ".jsonl":
        return read_jsonl(in_path)
    if suf == ".json":
        return read_json(in_path)
    if suf == ".csv":
        return read_csv(in_path)
    raise ValueError(f"Unsupported input format: {in_path}")

def main():
    ap = argparse.ArgumentParser(description="Generate eval_ground_truth.jsonl")
    ap.add_argument(
        "--in",
        dest="in_file",
        required=False,
        help="Input file (.jsonl/.json/.csv). If omitted, auto-detect from data/qa_pairs.",
    )
    ap.add_argument(
        "--out",
        dest="out_file",
        default="data/qa_pairs/eval_ground_truth.jsonl",
        help="Output JSONL path.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    qa_dir = root / "data" / "qa_pairs"

    if args.in_file:
        in_path = (root / args.in_file).resolve() if not Path(args.in_file).is_absolute() else Path(args.in_file)
    else:
        candidates = [
            qa_dir / "qa_pairs_eval.jsonl",
            qa_dir / "qa_pairs.jsonl",
            qa_dir / "eval_set.jsonl",
            qa_dir / "qa_pairs.json",
            qa_dir / "qa_pairs.csv",
        ]
        in_path = next((p for p in candidates if p.exists()), None)
        if in_path is None:
            raise FileNotFoundError(
                f"No input file found. Please pass --in. Checked: {[str(c) for c in candidates]}"
            )

    rows = load_rows(in_path)
    out_rows = []
    dropped = 0

    for i, r in enumerate(rows, 1):
        nr = norm_row(r, i)
        if nr is None:
            dropped += 1
            continue
        out_rows.append(nr)

    out_path = (root / args.out_file).resolve() if not Path(args.out_file).is_absolute() else Path(args.out_file)
    write_jsonl(out_path, out_rows)

    print(f"[OK] Input : {in_path}")
    print(f"[OK] Output: {out_path}")
    print(f"[OK] Kept={len(out_rows)}, Dropped={dropped}")

def main():
    ap = argparse.ArgumentParser(description="Generate eval_ground_truth.jsonl")
    ap.add_argument(
        "--in",
        dest="in_file",
        required=False,
        help="Input file (.jsonl/.json/.csv). If omitted, auto-detect from data/qa_pairs.",
    )
    ap.add_argument(
        "--out",
        dest="out_file",
        default="data/qa_pairs/eval_ground_truth.jsonl",
        help="Output JSONL path.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    qa_dir = root / "data" / "qa_pairs"

    if args.in_file:
        in_path = (root / args.in_file).resolve() if not Path(args.in_file).is_absolute() else Path(args.in_file)
    else:
        candidates = [
            qa_dir / "eval_seed.json",
            qa_dir / "qa_pairs_eval.jsonl",
            qa_dir / "qa_pairs.jsonl",
            qa_dir / "eval_set.jsonl",
            qa_dir / "qa_pairs.json",
            qa_dir / "qa_pairs.csv",
        ]
        in_path = next((p for p in candidates if p.exists()), None)
        if in_path is None:
            raise FileNotFoundError(
                f"No input file found. Please pass --in. Checked: {[str(c) for c in candidates]}"
            )

    rows = load_rows(in_path)
    out_rows = []
    dropped = 0

    for i, r in enumerate(rows, 1):
        nr = norm_row(r, i)
        if nr is None:
            dropped += 1
            continue
        out_rows.append(nr)

    out_path = (root / args.out_file).resolve() if not Path(args.out_file).is_absolute() else Path(args.out_file)
    write_jsonl(out_path, out_rows)

    print(f"[OK] Input : {in_path}")
    print(f"[OK] Output: {out_path}")
    print(f"[OK] Kept={len(out_rows)}, Dropped={dropped}")


if __name__ == "__main__":
    main()