import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[2]

def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)

def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    p_cnt: Dict[str, int] = {}
    g_cnt: Dict[str, int] = {}

    for t in p:
        p_cnt[t] = p_cnt.get(t, 0) + 1
    for t in g:
        g_cnt[t] = g_cnt.get(t, 0) + 1

    common = sum(min(c, g_cnt.get(t, 0)) for t, c in p_cnt.items())
    if common == 0:
        return 0.0

    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def citation_hit_rate(pred_citations: List[str], gold_citations: List[str]) -> float:
    if not gold_citations:
        return 1.0
    p = [normalize_text(x) for x in pred_citations if normalize_text(x)]
    g = [normalize_text(x) for x in gold_citations if normalize_text(x)]
    if not g:
        return 1.0
    if not p:
        return 0.0

    hit_count = 0
    for gold_item in g:
        matched = any(
            (pred_item == gold_item)
            or (pred_item in gold_item)
            or (gold_item in pred_item)
            for pred_item in p
        )
        if matched:
            hit_count += 1

    return hit_count / len(g)


def get_gold_citations(row: Dict) -> List[str]:
    """Prefer explicit gold_citations; fallback to references for legacy GT schema."""
    gold = row.get("gold_citations", [])
    if isinstance(gold, list) and gold:
        return gold
    refs = row.get("references", [])
    if isinstance(refs, list):
        return refs
    return []


def _normalize_items(values: Sequence[str]) -> List[str]:
    items: List[str] = []
    for value in values:
        if value is None:
            continue
        text = normalize_text(str(value))
        if text:
            items.append(text)
    return items


def get_gold_targets(row: Dict) -> List[str]:
    for key in ("gold_doc_ids", "references", "gold_citations"):
        values = row.get(key, [])
        if isinstance(values, list) and values:
            return _normalize_items(values)
    return []


def get_pred_targets(row: Dict, k: int) -> List[str]:
    for key in ("retrieved_doc_ids", "pred_citations"):
        values = row.get(key, [])
        if isinstance(values, list) and values:
            return _normalize_items(values[:k])
    return []


def _target_match(pred_item: str, gold_item: str) -> bool:
    return (
        pred_item == gold_item
        or pred_item in gold_item
        or gold_item in pred_item
    )


def _is_relevant(pred_item: str, gold_docs: List[str]) -> bool:
    return any(_target_match(pred_item, gold_item) for gold_item in gold_docs)


def _matched_gold_count(pred_docs: List[str], gold_docs: List[str]) -> int:
    matched_idx = set()
    for pred_item in pred_docs:
        for idx, gold_item in enumerate(gold_docs):
            if _target_match(pred_item, gold_item):
                matched_idx.add(idx)
    return len(matched_idx)


def recall_at_k(pred_docs: List[str], gold_docs: List[str]) -> float:
    if not gold_docs:
        return 0.0
    return _matched_gold_count(pred_docs, gold_docs) / len(gold_docs)


def precision_at_k(pred_docs: List[str], gold_docs: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for doc in pred_docs[:k] if _is_relevant(doc, gold_docs))
    return hits / k


def reciprocal_rank_at_k(pred_docs: List[str], gold_docs: List[str], k: int) -> float:
    for rank, doc_id in enumerate(pred_docs[:k], start=1):
        if _is_relevant(doc_id, gold_docs):
            return 1.0 / rank
    return 0.0


def average_precision_at_k(pred_docs: List[str], gold_docs: List[str], k: int) -> float:
    if not gold_docs:
        return 0.0

    hits = 0
    total = 0.0
    seen = set()
    for rank, doc_id in enumerate(pred_docs[:k], start=1):
        if _is_relevant(doc_id, gold_docs) and doc_id not in seen:
            hits += 1
            seen.add(doc_id)
            total += hits / rank
    return total / len(gold_docs)


def ndcg_at_k(pred_docs: List[str], gold_docs: List[str], k: int) -> float:
    if not gold_docs:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(pred_docs[:k], start=1):
        if _is_relevant(doc_id, gold_docs):
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold_docs), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def compute_retrieval_metrics(gt_rows: List[Dict], pred_rows: List[Dict], k: int) -> Dict[str, float]:
    pred_map = {r["id"]: r for r in pred_rows}

    recalls = []
    precisions = []
    mrrs = []
    ndcgs = []
    maps = []

    for g in gt_rows:
        qid = g["id"]
        gold_docs = get_gold_targets(g)
        pred_docs = get_pred_targets(pred_map.get(qid, {}), k)

        recalls.append(recall_at_k(pred_docs, gold_docs))
        precisions.append(precision_at_k(pred_docs, gold_docs, k))
        mrrs.append(reciprocal_rank_at_k(pred_docs, gold_docs, k))
        ndcgs.append(ndcg_at_k(pred_docs, gold_docs, k))
        maps.append(average_precision_at_k(pred_docs, gold_docs, k))

    return {
        f"recall@{k}": mean(recalls) if recalls else 0.0,
        f"precision@{k}": mean(precisions) if precisions else 0.0,
        f"ndcg@{k}": mean(ndcgs) if ndcgs else 0.0,
        f"mrr@{k}": mean(mrrs) if mrrs else 0.0,
        f"map@{k}": mean(maps) if maps else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to ground truth jsonl")
    parser.add_argument("--pred", required=True, help="Path to predictions jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", required=True, help="Path to output eval json")
    args = parser.parse_args()

    gt_rows = read_jsonl(resolve_path(args.gt))
    pred_rows = read_jsonl(resolve_path(args.pred))
    pred_map = {r["id"]: r for r in pred_rows}

    # Retrieval metrics
    retrieval = compute_retrieval_metrics(gt_rows, pred_rows, args.k)

    # Generation metrics
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ems, f1s, rouges, cites = [], [], [], []

    for g in gt_rows:
        qid = g["id"]
        p = pred_map.get(qid, {})
        pred_answer = p.get("pred_answer", "")
        gold_answer = g.get("gold_answer", "")

        ems.append(exact_match(pred_answer, gold_answer))
        f1s.append(token_f1(pred_answer, gold_answer))
        rouges.append(scorer.score(gold_answer, pred_answer)["rougeL"].fmeasure)

        pred_citations = p.get("pred_citations", [])
        gold_citations = get_gold_citations(g)
        cites.append(citation_hit_rate(pred_citations, gold_citations))

    report = {
        "retrieval": retrieval,
        "generation": {
            "exact_match": mean(ems) if ems else 0.0,
            "token_f1": mean(f1s) if f1s else 0.0,
            "rougeL_f1": mean(rouges) if rouges else 0.0,
            "citation_hit_rate": mean(cites) if cites else 0.0,
        },
        "meta": {
            "num_questions": len(gt_rows),
            "k": args.k,
        },
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()