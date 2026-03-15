"""
Main QA Agent for legal and tax questions.
"""

import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SCORING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "if", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "will", "with", "would", "your", "you", "now", "current",
    "currently", "please", "about", "into", "than", "then", "than", "such"
}


@dataclass
class AgentResponse:
    """Response from QA agent."""
    query: str
    answer: str
    confidence_score: float
    sources: List[str]
    confidence_breakdown: Dict[str, float] = None
    legal_citations: List[str] = None
    is_valid: bool = True
    processing_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.confidence_breakdown is None:
            self.confidence_breakdown = {}
        if self.legal_citations is None:
            self.legal_citations = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class QAAgent:
    """Legal and tax QA agent with retrieval-augmented generation."""
    
    def __init__(self, retriever, llm_service, validator, config: Dict):
        """
        Initialize QA Agent.
        
        Args:
            retriever: Document retriever
            llm_service: LLM service for answer generation
            validator: Answer validator
            config: Configuration dictionary
        """
        self.retriever = retriever
        self.llm_service = llm_service
        self.validator = validator
        self.config = config
        self.query_history = []

    def _config_get(self, key: str, default=None):
        """Read config values with lowercase and uppercase compatibility."""
        return self.config.get(key, self.config.get(key.upper(), default))
    
    def process_query(self, query: str, 
                     company_info: Optional[Dict] = None,
                     return_context: bool = False) -> AgentResponse:
        """
        Process user query through agent pipeline.
        
        Args:
            query: User question
            company_info: Optional company information
            return_context: Whether to include context in response
        
        Returns:
            AgentResponse with answer and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query[:80]}...")
            
            # Classify query type
            query_type = self._classify_query(query)
            logger.info(f"Query type: {query_type}")
            
            # Retrieve documents
            documents = self.retriever.retrieve(
                query=query,
                top_k=self.config.get('retrieval_top_k', 5),
                query_type=query_type
            )
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Generate answer
            answer = self.llm_service.generate_answer(
                query=query,
                context=documents,
                company_info=company_info,
                query_type=query_type
            )
            logger.info(f"Answer generated (length: {len(answer)} chars)")
            
            # Extract citations
            citations = self._extract_citations(answer)

            # Calculate grounded confidence
            confidence, confidence_breakdown = self._calculate_confidence(
                query=query,
                answer=answer,
                documents=documents,
                citations=citations,
            )
            
            # Get sources
            sources = [self._extract_source_label(doc) for doc in documents]
            
            processing_time = time.time() - start_time
            
            # Create response
            response = AgentResponse(
                query=query,
                answer=answer,
                confidence_score=confidence,
                confidence_breakdown=confidence_breakdown,
                sources=sources,
                legal_citations=citations,
                is_valid=True,
                processing_time=processing_time
            )
            
            # Store in history
            self.query_history.append({
                'query': query,
                'response': asdict(response),
                'timestamp': datetime.now()
            })
            
            logger.info(f"Query processed (confidence: {confidence:.2%}, time: {processing_time:.2f}s)")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    def _extract_source_label(self, doc: Dict) -> str:
        """Best-effort source label extraction for CLI display."""
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        source = (
            doc.get("source")
            or metadata.get("Law")
            or metadata.get("source")
            or metadata.get("title")
        )

        if source:
            return str(source)

        source_file = doc.get("_source_file")
        if source_file:
            return Path(str(source_file)).stem

        return "Unknown"
    
    def _classify_query(self, query: str) -> str:
        """Classify query into: general, compliance, financial, tax."""
        query_lower = query.lower()
        
        # Tax-related keywords
        tax_keywords = ['tax', 'gst', 'iras', 'income tax']
        # Financial-related keywords
        financial_keywords = ['balance sheet', 'income statement', 'financial']
        # Compliance-related keywords
        compliance_keywords = ['deadline', 'file', 'report', 'comply']
        
        # Check if query contains tax keywords
        if any(kw in query_lower for kw in tax_keywords):
            return 'tax'
        # Check if query contains financial keywords
        elif any(kw in query_lower for kw in financial_keywords):
            return 'financial'
        # Check if query contains compliance keywords
        elif any(kw in query_lower for kw in compliance_keywords):
            return 'compliance'
        else:
            return 'general'
    
    def _calculate_confidence(
        self,
        query: str,
        answer: str,
        documents: List[Dict],
        citations: List[str],
    ) -> tuple[float, Dict[str, float]]:
        """Calculate grounded confidence from retrieval quality, answer consistency, and citation coverage."""
        retrieval_quality = self._calculate_retrieval_quality(documents)
        answer_consistency = self._calculate_answer_consistency(query, answer, documents)
        citation_coverage = self._calculate_citation_coverage(citations, documents)

        retrieval_weight = float(self._config_get("confidence_retrieval_weight", 0.35))
        consistency_weight = float(self._config_get("confidence_consistency_weight", 0.40))
        citation_weight = float(self._config_get("confidence_citation_weight", 0.25))
        total_weight = retrieval_weight + consistency_weight + citation_weight
        if total_weight <= 0:
            retrieval_weight, consistency_weight, citation_weight = 0.35, 0.40, 0.25
            total_weight = 1.0

        confidence = (
            retrieval_weight * retrieval_quality
            + consistency_weight * answer_consistency
            + citation_weight * citation_coverage
        ) / total_weight
        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            "Confidence breakdown | retrieval_quality=%.3f | answer_consistency=%.3f | citation_coverage=%.3f | final=%.3f",
            retrieval_quality,
            answer_consistency,
            citation_coverage,
            confidence,
        )
        return confidence, {
            "retrieval_quality": retrieval_quality,
            "answer_consistency": answer_consistency,
            "citation_coverage": citation_coverage,
            "final": confidence,
        }

    def _calculate_retrieval_quality(self, documents: List[Dict]) -> float:
        """Estimate how strong the retrieved support set is."""
        if not documents:
            return 0.0

        target_k = max(int(self._config_get("retrieval_top_k", 5)), 1)
        depth_score = min(len(documents) / target_k, 1.0)

        supported_docs = 0
        score_values: List[float] = []
        for doc in documents:
            content = self._extract_doc_text(doc)
            source = self._extract_source_label(doc)
            if content and source != "Unknown":
                supported_docs += 1

            for key in ("rerank_score", "retrieval_score", "score"):
                value = doc.get(key)
                if isinstance(value, (int, float)):
                    score_values.append(float(value))
                    break

        support_quality = supported_docs / len(documents)

        score_quality = 0.5
        if score_values:
            min_score = min(score_values)
            max_score = max(score_values)
            if abs(max_score - min_score) < 1e-8:
                score_quality = 0.8 if max_score > 0 else 0.2
            else:
                normalized = [(value - min_score) / (max_score - min_score) for value in score_values]
                score_quality = sum(normalized) / len(normalized)

        return max(0.0, min(1.0, 0.30 * depth_score + 0.35 * support_quality + 0.35 * score_quality))

    def _calculate_answer_consistency(self, query: str, answer: str, documents: List[Dict]) -> float:
        """Estimate whether answer claims are supported by specific retrieved chunks and aligned with the query."""
        if not answer:
            return 0.0

        answer_tokens = self._tokenize_for_scoring(answer)
        query_tokens = self._tokenize_for_scoring(query)
        context_text = " ".join(self._extract_doc_text(doc) for doc in documents)
        context_tokens = set(self._tokenize_for_scoring(context_text))
        citations = self._extract_citations(answer)

        lexical_support = 0.0
        if answer_tokens and context_tokens:
            supported_answer_tokens = sum(1 for token in answer_tokens if token in context_tokens)
            lexical_support = supported_answer_tokens / len(answer_tokens)

        query_alignment = 0.0
        if query_tokens:
            answer_token_set = set(answer_tokens)
            matched_query_tokens = sum(1 for token in query_tokens if token in answer_token_set)
            query_alignment = matched_query_tokens / len(query_tokens)

        chunk_support = self._calculate_chunk_support(answer, documents)
        citation_sentence_support = self._calculate_citation_sentence_support(answer, citations, documents)

        validator_signal = 0.5
        if self.validator is not None:
            try:
                validation = self.validator.validate_answer(answer=answer, context=documents, query=query)
                validator_signal = float(validation.get("confidence", 0.5))
            except Exception as exc:
                logger.debug("Validator confidence unavailable: %s", exc)

        uncertainty_markers = ("might", "may", "unclear", "possibly", "could", "likely")
        uncertainty_penalty = 0.15 if any(marker in answer.lower() for marker in uncertainty_markers) else 0.0

        score = (
            0.15 * lexical_support
            + 0.15 * query_alignment
            + 0.35 * chunk_support
            + 0.20 * citation_sentence_support
            + 0.15 * validator_signal
            - uncertainty_penalty
        )
        return max(0.0, min(1.0, score))

    def _calculate_citation_coverage(self, citations: List[str], documents: List[Dict]) -> float:
        """Estimate how well answer citations are grounded in retrieved context."""
        if not documents:
            return 0.0
        if not citations:
            return 0.15

        supported = sum(1 for citation in citations if self._citation_supported(citation, documents))
        return supported / max(len(citations), 1)

    def _citation_supported(self, citation: str, documents: List[Dict]) -> bool:
        """Check whether a citation is supported by retrieved documents or source labels."""
        citation_norm = self._normalize_for_matching(citation)
        combined_context = self._normalize_for_matching(" ".join(self._extract_doc_text(doc) for doc in documents))
        source_labels = [self._normalize_for_matching(self._extract_source_label(doc)) for doc in documents]

        if " of the " in citation_norm and citation_norm.startswith("section "):
            section_part, act_part = citation_norm.split(" of the ", 1)
            section_match = re.search(r"section\s+([0-9a-z()]+)", section_part)
            section_id = section_match.group(1) if section_match else ""
            section_supported = bool(section_id) and re.search(rf"\b{re.escape(section_id)}\b", combined_context) is not None
            act_supported = act_part in combined_context or any(act_part in source for source in source_labels)
            return section_supported and act_supported

        if citation_norm.startswith("section ") or citation_norm.startswith("s. "):
            section_match = re.search(r"(?:section|s\.)\s+([0-9a-z()]+)", citation_norm)
            section_id = section_match.group(1) if section_match else ""
            return bool(section_id) and re.search(rf"\b{re.escape(section_id)}\b", combined_context) is not None

        return citation_norm in combined_context or any(citation_norm in source for source in source_labels)

    def _extract_doc_text(self, doc: Dict) -> str:
        """Extract main textual content from a retrieved document."""
        for key in ("content", "page_content", "text", "chunk", "body"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _tokenize_for_scoring(self, text: str) -> List[str]:
        """Tokenize text for lightweight overlap-based scoring."""
        tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        return [token for token in tokens if len(token) > 2 and token not in SCORING_STOPWORDS]

    def _split_answer_units(self, answer: str) -> List[str]:
        """Split answer into candidate evidence-bearing sentences or bullet units."""
        units: List[str] = []
        for block in re.split(r"\n+", answer or ""):
            block = re.sub(r"^[\-•*#\s]+", "", block).strip()
            if not block:
                continue
            units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", block) if part.strip())
        return units

    def _sentence_doc_support(self, sentence: str, doc: Dict) -> float:
        """Estimate support of one answer sentence by one retrieved chunk."""
        sentence_tokens = self._tokenize_for_scoring(sentence)
        if not sentence_tokens:
            return 0.0

        doc_tokens = set(self._tokenize_for_scoring(self._extract_doc_text(doc)))
        if not doc_tokens:
            return 0.0

        matched = sum(1 for token in sentence_tokens if token in doc_tokens)
        return matched / len(sentence_tokens)

    def _calculate_chunk_support(self, answer: str, documents: List[Dict]) -> float:
        """Measure whether answer claims are supported by at least one retrieved chunk."""
        if not documents:
            return 0.0

        threshold = float(self._config_get("confidence_sentence_support_threshold", 0.20))
        max_units = int(self._config_get("confidence_max_evidence_sentences", 8))
        units = [unit for unit in self._split_answer_units(answer) if len(self._tokenize_for_scoring(unit)) >= 3]
        if not units:
            return 0.0

        scored_units = []
        for unit in units[:max_units]:
            best_support = max((self._sentence_doc_support(unit, doc) for doc in documents), default=0.0)
            scored_units.append(best_support)

        strong_support_ratio = sum(1 for score in scored_units if score >= threshold) / len(scored_units)
        average_best_support = sum(scored_units) / len(scored_units)
        return max(0.0, min(1.0, 0.60 * strong_support_ratio + 0.40 * average_best_support))

    def _calculate_citation_sentence_support(self, answer: str, citations: List[str], documents: List[Dict]) -> float:
        """Measure whether sentences containing citations are supported by specific chunks containing those citations."""
        if not citations or not documents:
            return 0.0

        units = self._split_answer_units(answer)
        citation_support_scores: List[float] = []
        for citation in citations:
            citation_norm = self._normalize_for_matching(citation)
            containing_units = [unit for unit in units if citation_norm in self._normalize_for_matching(unit)]
            if not containing_units:
                citation_support_scores.append(0.0)
                continue

            best_citation_score = 0.0
            for unit in containing_units:
                best_doc_score = 0.0
                for doc in documents:
                    if not self._citation_supported(citation, [doc]):
                        continue
                    support = self._sentence_doc_support(unit, doc)
                    if support > best_doc_score:
                        best_doc_score = support
                best_citation_score = max(best_citation_score, best_doc_score)

            citation_support_scores.append(best_citation_score)

        return sum(citation_support_scores) / len(citation_support_scores)

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for citation support checks."""
        return re.sub(r"\s+", " ", (text or "").lower()).strip()
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text."""
        import re
        if not text:
            return []

        act_title = r'(?:[A-Z][A-Za-z0-9&()\-]*)(?:\s+[A-Z][A-Za-z0-9&()\-]*)*\s(?:Act|Regulation|Regulations)\s\d{4}'

        patterns = [
            rf'\b(?:Section|S\.)\s*\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*\s+of\s+the\s+{act_title}\b',
            rf'\b{act_title}\b',
            r'\b(?:Section|S\.)\s*\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*\b',
        ]

        raw_citations: List[str] = []
        seen = set()
        for i, pattern in enumerate(patterns):
            # Act-title patterns must NOT use IGNORECASE:
            # [A-Z] must stay strict so lowercase connective words like "the",
            # "based", "drawing" cannot anchor the start of a match.
            # Only the standalone Section pattern (index 2) uses IGNORECASE.
            flags = re.IGNORECASE if i == 2 else 0
            for match in re.finditer(pattern, text, flags):
                citation = re.sub(r'\s+', ' ', match.group(0)).strip(' ,.;:')
                key = citation.lower()
                if not citation or key in seen:
                    continue
                seen.add(key)
                raw_citations.append(citation)

        # Keep the most specific citation when shorter ones are contained inside it.
        raw_citations.sort(key=len, reverse=True)
        filtered: List[str] = []
        filtered_lower: List[str] = []
        for citation in raw_citations:
            citation_lower = citation.lower()
            # Must start with an uppercase letter (proper act name) or "Section/S."
            if not (citation[0].isupper() or citation_lower.startswith('section') or citation_lower.startswith('s.')):
                continue
            # Drop fragments that are clearly mid-sentence (common connective prefixes)
            connective_prefixes = ('of the ', 'and the ', 'or the ', 'in the ', 'under the ', 'pursuant to the ')
            if any(citation_lower.startswith(p) for p in connective_prefixes):
                continue
            # Drop if already fully contained within a longer citation already kept
            if any(citation_lower in kept for kept in filtered_lower):
                continue
            filtered.append(citation)
            filtered_lower.append(citation_lower)

        return filtered
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.query_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.query_history.clear()
        logger.info("Conversation history cleared")
