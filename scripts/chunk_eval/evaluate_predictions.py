import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List

from ranx import Qrels, Run, evaluate
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
    p = {normalize_text(x) for x in pred_citations}
    g = {normalize_text(x) for x in gold_citations}
    return len(p & g) / len(g) if g else 1.0


def build_qrels_and_run(gt_rows: List[Dict], pred_rows: List[Dict], k: int):
    pred_map = {r["id"]: r for r in pred_rows}
    qrels_dict = {}
    run_dict = {}

    for g in gt_rows:
        qid = g["id"]
        gold_docs = g.get("gold_doc_ids", [])
        pred_docs = pred_map.get(qid, {}).get("retrieved_doc_ids", [])[:k]

        qrels_dict[qid] = {doc_id: 1 for doc_id in gold_docs}
        run_dict[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(pred_docs)}

    return Qrels(qrels_dict), Run(run_dict)


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
    qrels, run = build_qrels_and_run(gt_rows, pred_rows, args.k)
    retrieval = evaluate(
        qrels,
        run,
        metrics=[
            f"recall@{args.k}",
            f"precision@{args.k}",
            f"ndcg@{args.k}",
            f"mrr@{args.k}",
            f"map@{args.k}",
        ],
    )

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
        gold_citations = g.get("gold_citations", [])
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