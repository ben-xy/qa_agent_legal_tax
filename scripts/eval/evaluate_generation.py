import argparse
import json
import re
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List

from rouge_score import rouge_scorer


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def em(pred: str, gold: str) -> float:
    return 1.0 if norm(pred) == norm(gold) else 0.0


def f1(pred: str, gold: str) -> float:
    p = norm(pred).split()
    g = norm(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_count, g_count = {}, {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1
    for t in g:
        g_count[t] = g_count.get(t, 0) + 1
    common = sum(min(c, g_count.get(t, 0)) for t, c in p_count.items())
    if common == 0:
        return 0.0
    pr = common / len(p)
    rc = common / len(g)
    return 2 * pr * rc / (pr + rc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=os.getenv("EVAL_GT"))
    parser.add_argument("--pred", default=os.getenv("EVAL_PRED"))
    parser.add_argument("--out", default=os.getenv("EVAL_OUT"))
    args = parser.parse_args()

    if not args.gt or not args.pred or not args.out:
        parser.error("the following arguments are required: --gt, --pred, --out")

    ROOT = Path(__file__).resolve().parents[2]

    def resolve_path(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (ROOT / path)

    gt_rows = read_jsonl(resolve_path(args.gt))
    pred_rows = read_jsonl(resolve_path(args.pred))
    pred_map = {r["id"]: r for r in pred_rows}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    em_list, f1_list, rouge_list = [], [], []
    for g in gt_rows:
        qid = g["id"]
        pred = pred_map.get(qid, {}).get("pred_answer", "")
        gold = g.get("gold_answer", "")

        em_list.append(em(pred, gold))
        f1_list.append(f1(pred, gold))
        rouge_list.append(scorer.score(gold, pred)["rougeL"].fmeasure)

    report = {
        "num_questions": len(gt_rows),
        "exact_match": mean(em_list) if em_list else 0.0,
        "token_f1": mean(f1_list) if f1_list else 0.0,
        "rougeL_f1": mean(rouge_list) if rouge_list else 0.0,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()