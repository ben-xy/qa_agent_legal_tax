import os
import sys
import json
import re
import inspect
import importlib
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import argparse

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import display_path


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


def _try_make(module_name: str, attr_name: str, cfg, cfg_dict):
    try:
        mod = importlib.import_module(module_name)
        obj = getattr(mod, attr_name)
        # class or factory
        if inspect.isclass(obj):
            for arg in (cfg_dict, cfg):
                try:
                    return obj(arg)
                except TypeError:
                    pass
            return obj()
        else:
            for arg in (cfg_dict, cfg):
                try:
                    return obj(arg)
                except TypeError:
                    pass
            return obj()
    except Exception:
        return None


def build_agent():
    from config import get_config
    from src.agents.qa_agent import QAAgent

    cfg = get_config()
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg

    print(f"QAAgent.__init__ signature: {inspect.signature(QAAgent.__init__)}")
    if isinstance(cfg_dict, dict):
        print(f"cfg_dict keys ({len(cfg_dict)}): {sorted(cfg_dict.keys())}")
    else:
        print(f"cfg_dict is not dict, type={type(cfg_dict).__name__}")

    retriever = (
        _try_make("src.retrievers.hybrid_retriever", "HybridRetriever", cfg, cfg_dict)
    )

    llm_service = (
        _try_make("src.services.llm_service", "LLMService", cfg, cfg_dict)
        or _try_make("src.llm.llm_service", "LLMService", cfg, cfg_dict)
        or _try_make("src.services.llm_service", "build_llm_service", cfg, cfg_dict)
    )

    validator = (
        _try_make("src.validators.legal_validator", "LegalValidator", cfg, cfg_dict)
    )

    if not (retriever and llm_service and validator):
        raise RuntimeError(
            f"Dependency build failed: retriever={type(retriever).__name__ if retriever else None}, "
            f"llm_service={type(llm_service).__name__ if llm_service else None}, "
            f"validator={type(validator).__name__ if validator else None}. "
            "Use rg result to replace module/class names with your real ones."
        )

    return QAAgent(
        retriever=retriever,
        llm_service=llm_service,
        validator=validator,
        config=cfg_dict,
    )


def _invoke_callable(fn, question: str, **extra_kwargs):
    """
    Call fn with best-effort signature matching.
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    # Prefer named semantic args
    for key in ("query", "question", "text", "prompt", "input"):
        if key in params:
            kwargs = {key: question}
            for k, v in extra_kwargs.items():
                if k in params:
                    kwargs[k] = v
            return fn(**kwargs)

    # Fallback: first positional argument
    kwargs = {k: v for k, v in extra_kwargs.items() if k in params}
    return fn(question, **kwargs)


def _manual_agent_fallback(agent: Any, question: str) -> Dict[str, Any]:
    """
    Fallback pipeline based on provided components:
    retriever -> llm_service -> validator
    """
    retriever = getattr(agent, "retriever", None)
    llm_service = getattr(agent, "llm_service", None)
    validator = getattr(agent, "validator", None)
    config = getattr(agent, "config", {}) or {}

    if not retriever or not llm_service:
        raise RuntimeError("Agent fallback failed: missing retriever or llm_service.")

    top_k = int(config.get("RETRIEVAL_TOP_K", config.get("retrieval_top_k", 5)))
    query_type = "general"

    # retrieve
    try:
        contexts = retriever.retrieve(query=question, top_k=top_k, query_type=query_type)
    except TypeError:
        try:
            contexts = retriever.retrieve(question, top_k=top_k, query_type=query_type)
        except TypeError:
            contexts = retriever.retrieve(question)

    # generate
    answer = llm_service.generate_answer(
        query=question,
        context=contexts,
        company_info=None,
        query_type=query_type,
    )

    # validate + citations
    validation = None
    citations: List[str] = []
    if validator:
        if hasattr(validator, "validate_answer"):
            validation = validator.validate_answer(answer=answer, context=contexts, query=question)
        if hasattr(validator, "extract_citations"):
            citations = validator.extract_citations(answer) or []

    return {
        "answer": answer,
        "documents": contexts,
        "citations": citations,
        "validation": validation,
    }


def call_agent(agent: Any, question: str) -> Any:
    """
    Robust call order:
    1) Try common QAAgent methods
    2) Fallback to manual component pipeline
    """
    candidate_methods = [
        "answer_question",
        "answer",
        "ask",
        "query",
        "run",
        "invoke",
        "chat",
        "generate",
        "process_query",
    ]

    for method_name in candidate_methods:
        method = getattr(agent, method_name, None)
        if callable(method):
            try:
                return _invoke_callable(method, question)
            except TypeError:
                continue

    if callable(agent):
        try:
            return _invoke_callable(agent, question)
        except TypeError:
            pass

    return _manual_agent_fallback(agent, question)


def normalize_response(resp: Any) -> Dict[str, Any]:
    """
        Convert agent response into unified schema:
    {
      pred_answer: str,
      retrieved_doc_ids: List[str],
            eval_friendly_doc_ids: List[str],
      pred_citations: List[str]
    }
    """

    def _normalize_text(v: Any) -> str:
        if v is None:
            return ""
        return " ".join(str(v).strip().split())

    def _stable_chunk_id(doc: Dict[str, Any]) -> str:
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}

        source = (
            doc.get("source")
            or metadata.get("source")
            or metadata.get("Law")
            or metadata.get("title")
            or doc.get("_source_file")
            or "unknown"
        )

        # Prefer explicit chunk location hints when available.
        chunk_part = (
            doc.get("chunk_id")
            or doc.get("chunk_index")
            or metadata.get("chunk_id")
            or metadata.get("chunk_index")
            or metadata.get("index")
            or metadata.get("page")
            or metadata.get("start")
            or metadata.get("line_start")
            or "na"
        )

        text = (
            doc.get("content")
            or doc.get("page_content")
            or doc.get("text")
            or doc.get("chunk")
            or ""
        )
        digest = hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()[:12]
        return f"{source}::chunk={chunk_part}::h={digest}"

    def _extract_law_title(raw: Any) -> str:
        if raw is None:
            return ""

        text = " ".join(str(raw).replace("\n", " ").split())
        if not text:
            return ""

        candidates = [text]
        if "::" in text:
            candidates.extend([part.strip() for part in text.split("::") if part.strip()])

        for token in (" - Singapore Statutes Online", " > ", "|"):
            extra = []
            for c in candidates:
                if token in c:
                    extra.append(c.split(token)[0].strip())
            candidates.extend([e for e in extra if e])

        law_pattern = re.compile(
            r"([A-Za-z][A-Za-z0-9'()\-/\s]*?\b(?:Act|Regulations?|Rules?|Code|Order|Ordinance|Constitution|Charter)\b(?:\s*\d{4})?)",
            re.IGNORECASE,
        )

        best = ""
        for candidate in candidates:
            match = law_pattern.search(candidate)
            if match:
                current = " ".join(match.group(1).split())
                if len(current) > len(best):
                    best = current
        if best:
            return best

        fallback = candidates[0]
        fallback = re.sub(r"^doc[_-]?\d+\s*[:\-]*\s*", "", fallback, flags=re.IGNORECASE)
        return " ".join(fallback.split())

    def _canonical_doc_id(doc: Dict[str, Any]) -> str:
        explicit = doc.get("_doc_id") or doc.get("id") or doc.get("doc_id")
        chunk_id = _stable_chunk_id(doc)

        if explicit is None:
            return chunk_id

        explicit_text = str(explicit).strip()
        if not explicit_text:
            return chunk_id

        # Keep already-granular IDs; otherwise append chunk/hash to avoid coarse law-level collisions.
        coarse_markers = ("::chunk=", "#chunk", "chunk=", "@chunk", ":chunk")
        if any(marker in explicit_text.lower() for marker in coarse_markers):
            return explicit_text

        return f"{explicit_text}::{chunk_id}"

    if isinstance(resp, str):
        return {
            "pred_answer": resp,
            "retrieved_doc_ids": [],
            "eval_friendly_doc_ids": [],
            "pred_citations": [],
        }

    if hasattr(resp, "answer") or hasattr(resp, "response"):
        answer = getattr(resp, "answer", None) or getattr(resp, "response", None) or getattr(resp, "final_answer", None) or ""
        citations = (
            getattr(resp, "citations", None)
            or getattr(resp, "legal_citations", None)
            or []
        )
        contexts = (
            getattr(resp, "contexts", None)
            or getattr(resp, "documents", None)
            or getattr(resp, "retrieved_docs", None)
            or getattr(resp, "sources", None)
            or []
        )

        retrieved_doc_ids: List[str] = []
        eval_friendly_doc_ids: List[str] = []
        for c in contexts:
            if isinstance(c, dict):
                doc_id = _canonical_doc_id(c)
                metadata = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
                eval_title = (
                    _extract_law_title(c.get("law_title"))
                    or _extract_law_title(c.get("source"))
                    or _extract_law_title(metadata.get("Law"))
                    or _extract_law_title(metadata.get("title"))
                    or _extract_law_title(doc_id)
                )
            elif isinstance(c, str):
                doc_id = c.strip()
                eval_title = _extract_law_title(c)
            else:
                doc_id = getattr(c, "_doc_id", None) or getattr(c, "id", None) or getattr(c, "doc_id", None)
                eval_title = _extract_law_title(doc_id)
            if doc_id is not None:
                retrieved_doc_ids.append(str(doc_id))
            if eval_title:
                eval_friendly_doc_ids.append(eval_title)

        if citations and isinstance(citations, list) and not isinstance(citations[0], str):
            citations = [
                getattr(x, "text", None)
                or getattr(x, "citation", None)
                or (x.get("text") if isinstance(x, dict) else None)
                or (x.get("citation") if isinstance(x, dict) else None)
                or str(x)
                for x in citations
            ]

        return {
            "pred_answer": str(answer),
            "retrieved_doc_ids": retrieved_doc_ids,
            "eval_friendly_doc_ids": eval_friendly_doc_ids,
            "pred_citations": citations if isinstance(citations, list) else [],
        }

    if isinstance(resp, dict):
        answer = resp.get("answer") or resp.get("final_answer") or resp.get("response") or ""
        citations = resp.get("citations") or []
        contexts = resp.get("contexts") or resp.get("documents") or resp.get("retrieved_docs") or []

        retrieved_doc_ids: List[str] = []
        eval_friendly_doc_ids: List[str] = []
        for c in contexts:
            if isinstance(c, dict):
                doc_id = _canonical_doc_id(c)
                metadata = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
                eval_title = (
                    _extract_law_title(c.get("law_title"))
                    or _extract_law_title(c.get("source"))
                    or _extract_law_title(metadata.get("Law"))
                    or _extract_law_title(metadata.get("title"))
                    or _extract_law_title(doc_id)
                )
                if doc_id is not None:
                    retrieved_doc_ids.append(str(doc_id))
                if eval_title:
                    eval_friendly_doc_ids.append(eval_title)
            elif isinstance(c, str):
                text = c.strip()
                if text:
                    retrieved_doc_ids.append(text)
                    eval_title = _extract_law_title(text)
                    if eval_title:
                        eval_friendly_doc_ids.append(eval_title)

        if citations and isinstance(citations, list) and isinstance(citations[0], dict):
            citations = [x.get("text") or x.get("citation") or str(x) for x in citations]

        return {
            "pred_answer": answer,
            "retrieved_doc_ids": retrieved_doc_ids,
            "eval_friendly_doc_ids": eval_friendly_doc_ids,
            "pred_citations": citations if isinstance(citations, list) else [],
        }

    return {
        "pred_answer": str(resp),
        "retrieved_doc_ids": [],
        "eval_friendly_doc_ids": [],
        "pred_citations": [],
    }


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_retrieval_only_prediction(agent: Any, question: str, top_k: int) -> Dict[str, Any]:
    retriever = getattr(agent, "retriever", None)
    if retriever is None:
        raise RuntimeError("retrieval-only mode requires agent.retriever, but it was not found.")

    # Retrieval-only benchmarking should not invoke external rerank APIs.
    if hasattr(retriever, "enable_rerank"):
        try:
            retriever.enable_rerank = False
        except Exception:
            pass

    query_type = "general"
    try:
        contexts = retriever.retrieve(query=question, top_k=top_k, query_type=query_type)
    except TypeError:
        try:
            contexts = retriever.retrieve(question, top_k=top_k, query_type=query_type)
        except TypeError:
            contexts = retriever.retrieve(question)

    return {
        "answer": "",
        "documents": contexts,
        "citations": [],
    }


def _write_run_snapshot(args, pred_out_path: Path) -> Path:
    """
    Write a reproducibility snapshot with fixed filename pattern:
    outputs/run_snapshot_YYYYmmdd_HHMMSS.json
    """
    outputs_dir = pred_out_path.parent
    outputs_dir.mkdir(parents=True, exist_ok=True)

    top_k = getattr(args, "top_k", None) or getattr(args, "k", None) or 5
    enable_rerank = _to_bool(getattr(args, "enable_rerank", None))
    rerank_model = (
        getattr(args, "rerank_model", None)
        or os.getenv("COHERE_RERANK_MODEL")
        or "rerank-english-v3.0"
    )
    llm_model = (
        getattr(args, "model", None)
        or getattr(args, "llm_model", None)
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL_NAME")
        or ""
    )

    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    snapshot_path = outputs_dir / f"run_snapshot_{ts}.json"

    snapshot = {
        "timestamp_local": now.isoformat(timespec="seconds"),
        "ground_truth": display_path(resolve_path(getattr(args, "gt", "")), ROOT),
        "pred_output": display_path(pred_out_path, ROOT),
        "settings": {
            "enable_rerank": enable_rerank,
            "top_k": int(top_k),
            "llm_model": llm_model,
            "rerank_model": rerank_model,
        },
    }

    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[snapshot] Saved run config snapshot: {display_path(snapshot_path, ROOT)}")
    return snapshot_path


def main():
    parser = argparse.ArgumentParser()
    # default paths, will be replaced with actual ones in eval script calls
    parser.add_argument(
        "--gt",
        default="data/qa_pairs/eval_ground_truth.jsonl",
        help="Path to ground truth jsonl",
    )
    parser.add_argument(
        "--out",
        default="outputs/preds_hybrid.jsonl",
        help="Path to output predictions jsonl",
    )
    parser.add_argument(
        "--enable-rerank",
        choices=["true", "false"],
        default="false",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K docs used for answering/eval snapshot")
    parser.add_argument("--model", type=str, default=None, help="LLM model name for snapshot only")
    parser.add_argument("--rerank-model", type=str, default=None, help="Rerank model name for snapshot only")
    parser.add_argument(
        "--retrieval-only",
        choices=["true", "false"],
        default="false",
        help="If true, skip LLM generation and run retrieval-only predictions.",
    )
    args = parser.parse_args()

    os.environ["ENABLE_RERANK"] = args.enable_rerank

    gt_path = resolve_path(args.gt)
    out_path = resolve_path(args.out)

    gt_rows = read_jsonl(gt_path)
    agent = build_agent()
    retrieval_only = _to_bool(args.retrieval_only)

    preds: List[Dict[str, Any]] = []
    for row in gt_rows:
        qid = row["id"]
        question = row["question"]
        if retrieval_only:
            resp = build_retrieval_only_prediction(agent, question, args.top_k)
        else:
            resp = call_agent(agent, question)
        norm = normalize_response(resp)

        preds.append(
            {
                "id": qid,
                "pred_answer": norm["pred_answer"],
                "retrieved_doc_ids": norm["retrieved_doc_ids"],
                "eval_friendly_doc_ids": norm["eval_friendly_doc_ids"],
                "pred_citations": norm["pred_citations"],
            }
        )

    write_jsonl(out_path, preds)
    print(f"Saved predictions: {display_path(out_path, ROOT)} (n={len(preds)})")
    _write_run_snapshot(args, out_path)


if __name__ == "__main__":
    main()