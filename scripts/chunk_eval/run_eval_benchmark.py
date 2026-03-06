import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_agent():
    from config import get_config
    from src.agents.qa_agent import QAAgent

    cfg = get_config()
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
    return QAAgent(cfg_dict)


def call_agent(agent: Any, question: str) -> Any:
    """
    Fallback for different agent APIs in projects.
    """
    for method_name in ["answer_question", "ask", "query", "run", "__call__"]:
        if hasattr(agent, method_name):
            method = getattr(agent, method_name)
            return method(question)
    raise RuntimeError("No supported QAAgent method found (answer_question/ask/query/run/__call__).")


def normalize_response(resp: Any) -> Dict[str, Any]:
    """
    Convert agent response into unified schema:
    {
      pred_answer: str,
      retrieved_doc_ids: List[str],
      pred_citations: List[str]
    }
    """
    if isinstance(resp, str):
        return {
            "pred_answer": resp,
            "retrieved_doc_ids": [],
            "pred_citations": [],
        }

    if isinstance(resp, dict):
        answer = resp.get("answer") or resp.get("final_answer") or resp.get("response") or ""
        citations = resp.get("citations") or []
        contexts = resp.get("contexts") or resp.get("documents") or resp.get("retrieved_docs") or []

        retrieved_doc_ids: List[str] = []
        for c in contexts:
            if isinstance(c, dict):
                doc_id = c.get("_doc_id") or c.get("id") or c.get("doc_id")
                if doc_id is not None:
                    retrieved_doc_ids.append(str(doc_id))

        if citations and isinstance(citations, list) and isinstance(citations[0], dict):
            citations = [x.get("text") or x.get("citation") or str(x) for x in citations]

        return {
            "pred_answer": answer,
            "retrieved_doc_ids": retrieved_doc_ids,
            "pred_citations": citations if isinstance(citations, list) else [],
        }

    return {
        "pred_answer": str(resp),
        "retrieved_doc_ids": [],
        "pred_citations": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to ground truth jsonl")
    parser.add_argument("--out", required=True, help="Path to output predictions jsonl")
    parser.add_argument("--enable-rerank", choices=["true", "false"], required=True)
    args = parser.parse_args()

    os.environ["ENABLE_RERANK"] = args.enable_rerank

    gt_path = resolve_path(args.gt)
    out_path = resolve_path(args.out)

    gt_rows = read_jsonl(gt_path)
    agent = build_agent()

    preds: List[Dict[str, Any]] = []
    for row in gt_rows:
        qid = row["id"]
        question = row["question"]
        resp = call_agent(agent, question)
        norm = normalize_response(resp)

        preds.append(
            {
                "id": qid,
                "pred_answer": norm["pred_answer"],
                "retrieved_doc_ids": norm["retrieved_doc_ids"],
                "pred_citations": norm["pred_citations"],
            }
        )

    write_jsonl(out_path, preds)
    print(f"Saved predictions: {out_path} (n={len(preds)})")


if __name__ == "__main__":
    main()