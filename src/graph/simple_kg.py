"""
Minimal legal knowledge graph for retrieval-time boosting.

Graph nodes: legal entities (Act names, Section references)
Graph edges: entity co-occurrence within the same document chunk
"""

import re
from collections import defaultdict
from typing import Dict, List, Set


class SimpleLegalKG:
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.entity_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)
        self._build()

    def _get_text(self, doc: Dict) -> str:
        return (
            doc.get("content")
            or doc.get("text")
            or doc.get("chunk")
            or doc.get("body")
            or ""
        )

    def _get_doc_id(self, doc: Dict, fallback_idx: int) -> str:
        return str(doc.get("_doc_id") or doc.get("id") or f"doc_{fallback_idx}")

    def _extract_entities(self, text: str) -> Set[str]:
        entities = set()

        # Example: "Income Tax Act", "Goods and Services Tax Act"
        for m in re.finditer(r"\b([A-Z][A-Za-z&\-\s]{2,80} Act)\b", text):
            entities.add(m.group(1).strip().lower())

        # Example: "Section 10", "Section 10(1)", "s. 10(1)"
        for m in re.finditer(r"\b(?:section|s\.)\s*\d+[A-Za-z]?(?:\(\d+\))?\b", text, flags=re.IGNORECASE):
            entities.add(m.group(0).strip().lower())

        return entities

    def _build(self) -> None:
        for i, doc in enumerate(self.documents):
            doc_id = self._get_doc_id(doc, i)
            text = self._get_text(doc)
            entities = self._extract_entities(text)

            if not entities:
                continue

            for e in entities:
                self.entity_to_doc_ids[e].add(doc_id)

            entity_list = list(entities)
            for i1 in range(len(entity_list)):
                for i2 in range(i1 + 1, len(entity_list)):
                    a, b = entity_list[i1], entity_list[i2]
                    self.entity_graph[a].add(b)
                    self.entity_graph[b].add(a)

    def expand(self, query: str, max_docs: int = 50) -> Set[str]:
        """
        Return expanded related doc IDs based on:
        1) entities directly found in query
        2) one-hop neighboring entities
        """
        seed_entities = self._extract_entities(query)
        if not seed_entities:
            return set()

        expanded_entities = set(seed_entities)
        for e in list(seed_entities):
            expanded_entities.update(self.entity_graph.get(e, set()))

        doc_ids: Set[str] = set()
        for e in expanded_entities:
            doc_ids.update(self.entity_to_doc_ids.get(e, set()))
            if len(doc_ids) >= max_docs:
                break

        return set(list(doc_ids)[:max_docs])