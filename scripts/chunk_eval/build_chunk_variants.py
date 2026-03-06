import re
import json
from pathlib import Path
from typing import List, Dict


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "data" / "acts_md"
OUT_FIXED = ROOT / "data" / "acts_chunked_fixed"
OUT_STRUCT = ROOT / "data" / "acts_chunked_struct"

print(f"Source directory: {SRC_DIR}")
print(f"Fixed chunk output directory: {OUT_FIXED}")
print(f"Structure-aware chunk output directory: {OUT_STRUCT}")

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_fixed(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        if j >= n:
            break
        i = max(0, j - overlap)

    return [c for c in chunks if c.strip()]


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
    out: List[str] = []

    if not blocks:
        return chunk_fixed(text, chunk_size=fallback_size, overlap=overlap)

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
        rows.append(
            {
                "id": f"{path.stem}_{strategy}_{i}",
                "source_file": path.name,
                "chunk_index": i,
                "content": c,
            }
        )
    return rows


def run(strategy: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC_DIR.glob("*.md"))

    for f in files:
        rows = build_chunks_for_file(f, strategy)
        out_path = out_dir / f"{f.stem}.json"
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] strategy={strategy}, files={len(files)}, out={out_dir}")


if __name__ == "__main__":
    run("fixed", OUT_FIXED)
    run("struct", OUT_STRUCT)

import argparse
import json
from pathlib import Path

from src.utils.chunking_strategies import fixed_size_chunks, section_aware_chunks


def build_for_file(md_path: Path, strategy: str):
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    act_name = md_path.stem

    if strategy == "fixed":
        return fixed_size_chunks(text=text, chunk_size=1200, overlap=200, act_name=act_name)
    if strategy == "section":
        return section_aware_chunks(text=text, max_chunk_size=1600, overlap=150, act_name=act_name)

    raise ValueError(f"Unsupported strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/acts_md")
    parser.add_argument("--output_dir", default="data/acts_chunked_variants")
    parser.add_argument("--strategy", choices=["fixed", "section", "both"], default="both")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    strategies = ["fixed", "section"] if args.strategy == "both" else [args.strategy]

    for st in strategies:
        st_dir = output_dir / st
        st_dir.mkdir(parents=True, exist_ok=True)

        total_chunks = 0
        for md in md_files:
            chunks = build_for_file(md, st)
            total_chunks += len(chunks)
            out_path = st_dir / f"{md.stem}.json"
            out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{st}] files={len(md_files)}, total_chunks={total_chunks}, output={st_dir}")


if __name__ == "__main__":
    main()