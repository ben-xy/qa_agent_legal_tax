import re
from typing import Dict, List, Optional


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fixed_size_chunks(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    act_name: Optional[str] = None
) -> List[Dict]:
    text = _clean_text(text)
    chunks: List[Dict] = []
    if not text:
        return chunks

    step = max(1, chunk_size - overlap)
    n = len(text)
    idx = 0
    chunk_id = 0

    while idx < n:
        end = min(idx + chunk_size, n)
        piece = text[idx:end].strip()
        if piece:
            chunks.append(
                {
                    "chunk_id": f"{act_name or 'act'}::fixed::{chunk_id}",
                    "strategy": "fixed",
                    "act_name": act_name,
                    "content": piece,
                    "start_char": idx,
                    "end_char": end,
                }
            )
            chunk_id += 1
        if end == n:
            break
        idx += step

    return chunks


_HEADING_RE = re.compile(
    r"(?m)^(#{1,6}\s+.+)$|(?m)^(Section\s+\d+[A-Za-z]?(?:\([^)]+\))?.*)$",
    re.IGNORECASE,
)


def section_aware_chunks(
    text: str,
    max_chunk_size: int = 1600,
    overlap: int = 150,
    act_name: Optional[str] = None
) -> List[Dict]:
    text = _clean_text(text)
    chunks: List[Dict] = []
    if not text:
        return chunks

    # Split by heading-like boundaries while preserving heading lines.
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return fixed_size_chunks(
            text=text,
            chunk_size=max_chunk_size,
            overlap=overlap,
            act_name=act_name,
        )

    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((start, end, m.group(0).strip()))

    chunk_id = 0
    for start, end, heading in spans:
        block = text[start:end].strip()
        if not block:
            continue

        # If block is too long, fallback to fixed chunks within this section.
        if len(block) > max_chunk_size:
            sub = fixed_size_chunks(
                text=block,
                chunk_size=max_chunk_size,
                overlap=overlap,
                act_name=act_name,
            )
            for s in sub:
                s["strategy"] = "section"
                s["heading"] = heading
                s["chunk_id"] = f"{act_name or 'act'}::section::{chunk_id}"
                chunk_id += 1
                chunks.append(s)
        else:
            chunks.append(
                {
                    "chunk_id": f"{act_name or 'act'}::section::{chunk_id}",
                    "strategy": "section",
                    "act_name": act_name,
                    "heading": heading,
                    "content": block,
                    "start_char": start,
                    "end_char": end,
                }
            )
            chunk_id += 1

    return chunks