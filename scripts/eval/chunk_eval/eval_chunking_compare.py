import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def collect_metrics(eval_obj: Dict[str, Any]) -> Dict[str, float]:
    """
    Support format:
    {
      "retrieval": {"recall@5": ..., "precision@5": ..., ...},
      "generation": {"exact_match": ..., "token_f1": ..., ...}
    }
    """
    out: Dict[str, float] = {}

    retrieval = eval_obj.get("retrieval", {})
    generation = eval_obj.get("generation", {})

    # retrieval metrics: keep keys like recall@5, mrr@5, ndcg@5 ...
    for k, v in retrieval.items():
        out[f"retrieval.{k}"] = to_float(v)

    # generation metrics
    for k in ["exact_match", "token_f1", "rougeL_f1", "citation_hit_rate"]:
        if k in generation:
            out[f"generation.{k}"] = to_float(generation.get(k))

    return out


def union_metric_names(a: Dict[str, float], b: Dict[str, float]) -> List[str]:
    names = sorted(set(a.keys()) | set(b.keys()))
    # Put common report metrics first
    priority = [
        "retrieval.recall@5",
        "retrieval.precision@5",
        "retrieval.ndcg@5",
        "retrieval.mrr@5",
        "retrieval.map@5",
        "generation.exact_match",
        "generation.token_f1",
        "generation.rougeL_f1",
        "generation.citation_hit_rate",
    ]
    priority_set = set(priority)
    ordered = [x for x in priority if x in names]
    ordered += [x for x in names if x not in priority_set]
    return ordered


def build_rows(
    fixed_metrics: Dict[str, float],
    struct_metrics: Dict[str, float],
) -> List[Tuple[str, float, float, float, str]]:
    rows = []
    for name in union_metric_names(fixed_metrics, struct_metrics):
        f = fixed_metrics.get(name, 0.0)
        s = struct_metrics.get(name, 0.0)
        d = s - f
        if abs(d) < 1e-12:
            winner = "tie"
        else:
            winner = "struct" if d > 0 else "fixed"
        rows.append((name, f, s, d, winner))
    return rows


def fmt(x: float) -> str:
    return f"{x:.4f}"


def make_markdown(
    rows: List[Tuple[str, float, float, float, str]],
    fixed_name: str,
    struct_name: str,
    fixed_file: str,
    struct_file: str,
) -> str:
    lines: List[str] = []
    lines.append("# Chunking Strategy Evaluation Comparison")
    lines.append("")
    lines.append(f"- Fixed strategy file: `{fixed_file}`")
    lines.append(f"- Structure-aware strategy file: `{struct_file}`")
    lines.append("")
    lines.append(f"- Strategy A: **{fixed_name}**")
    lines.append(f"- Strategy B: **{struct_name}**")
    lines.append("")
    lines.append("| Metric | Fixed | Structure-aware | Delta (Struct - Fixed) | Better |")
    lines.append("|---|---:|---:|---:|---|")
    for name, f, s, d, winner in rows:
        sign = "+" if d >= 0 else ""
        lines.append(f"| {name} | {fmt(f)} | {fmt(s)} | {sign}{fmt(d)} | {winner} |")
    lines.append("")
    return "\n".join(lines)


def save_json_summary(path: Path, rows: List[Tuple[str, float, float, float, str]]) -> None:
    summary = {
        "comparison": [
            {
                "metric": name,
                "fixed": f,
                "structure_aware": s,
                "delta_struct_minus_fixed": d,
                "better": winner,
            }
            for name, f, s, d, winner in rows
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare chunking evaluation results (fixed vs structure-aware).")
    parser.add_argument("--fixed", required=True, help="Path to fixed chunking eval JSON")
    parser.add_argument("--struct", required=True, help="Path to structure-aware chunking eval JSON")
    parser.add_argument("--fixed-name", default="Fixed-size chunking")
    parser.add_argument("--struct-name", default="Structure-aware chunking")
    parser.add_argument("--out-md", default="outputs/eval_chunking_comparison.md")
    parser.add_argument("--out-json", default="outputs/eval_chunking_comparison.json")
    args = parser.parse_args()

    fixed_path = Path(args.fixed)
    struct_path = Path(args.struct)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)

    fixed_eval = load_json(fixed_path)
    struct_eval = load_json(struct_path)

    fixed_metrics = collect_metrics(fixed_eval)
    struct_metrics = collect_metrics(struct_eval)
    rows = build_rows(fixed_metrics, struct_metrics)

    md = make_markdown(
        rows=rows,
        fixed_name=args.fixed_name,
        struct_name=args.struct_name,
        fixed_file=str(fixed_path),
        struct_file=str(struct_path),
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    save_json_summary(out_json, rows)

    print(md)
    print(f"\nSaved: {out_md}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()