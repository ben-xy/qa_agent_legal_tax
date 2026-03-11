import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]  # repo root (fixed)


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def chunk_fixed(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - overlap)

    while i < n:
        j = min(i + chunk_size, n)
        piece = text[i:j].strip()
        if piece:
            chunks.append(piece)
        if j >= n:
            break
        i += step

    return chunks


def split_by_structure(text: str) -> List[str]:
    """
    Minimal legal structure split by common markers.
    """
    pattern = r"(?=\b(?:Section\s+\d+[A-Za-z]?(?:\(\d+\))?|PART\s+[IVXLC]+|CHAPTER\s+\d+)\b)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [normalize_text(p) for p in parts if p and p.strip()]


def chunk_structure_aware(
    text: str,
    max_len: int = 1200,
    fallback_size: int = 800,
    overlap: int = 120,
) -> List[str]:
    blocks = split_by_structure(text)
    if not blocks:
        return chunk_fixed(text, chunk_size=fallback_size, overlap=overlap)

    out: List[str] = []
    for b in blocks:
        if len(b) <= max_len:
            out.append(b)
        else:
            out.extend(chunk_fixed(b, chunk_size=fallback_size, overlap=overlap))
    return [c for c in out if c.strip()]


def build_chunks_for_file(path: Path, strategy: str) -> List[Dict]:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")

    if strategy == "fixed":
        chunks = chunk_fixed(raw_text)
    elif strategy == "struct":
        chunks = chunk_structure_aware(raw_text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rows: List[Dict] = []
    for i, c in enumerate(chunks):
        chunk_id = f"{path.stem}_{strategy}_{i}"
        rows.append(
            {
                "id": chunk_id,              # backward compatibility
                "chunk_id": chunk_id,        # preferred
                "source_file": path.name,
                "chunk_index": i,
                "strategy": strategy,
                "content": c,
            }
        )
    return rows


def run(strategy: str, src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {src_dir}")

    for f in files:
        rows = build_chunks_for_file(f, strategy)
        out_path = out_dir / f"{f.stem}.json"
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] strategy={strategy}, files={len(files)}, out={out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default=str(ROOT / "data" / "acts_md"))
    parser.add_argument("--out-fixed", default=str(ROOT / "data" / "tmp" / "acts_chunked_fixed"))
    parser.add_argument("--out-struct", default=str(ROOT / "data" / "tmp" / "acts_chunked_struct"))
    args = parser.parse_args()

    src_dir = resolve_path(args.src_dir)
    out_fixed = resolve_path(args.out_fixed)
    out_struct = resolve_path(args.out_struct)

    print(f"Source directory: {src_dir}")
    print(f"Fixed chunk output directory: {out_fixed}")
    print(f"Structure-aware chunk output directory: {out_struct}")

    run("fixed", src_dir, out_fixed)
    run("struct", src_dir, out_struct)


if __name__ == "__main__":
    main()