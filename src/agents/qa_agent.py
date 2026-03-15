"""
Main QA Agent for legal and tax questions.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from QA agent."""
    query: str
    answer: str
    confidence_score: float
    sources: List[str]
    legal_citations: List[str] = None
    is_valid: bool = True
    processing_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
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
            
            # Calculate confidence
            confidence = self._calculate_confidence(answer, len(documents))
            
            # Extract citations
            citations = self._extract_citations(answer)
            
            # Get sources
            sources = [self._extract_source_label(doc) for doc in documents]
            
            processing_time = time.time() - start_time
            
            # Create response
            response = AgentResponse(
                query=query,
                answer=answer,
                confidence_score=confidence,
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
    
    def _calculate_confidence(self, answer: str, doc_count: int) -> float:
        """Calculate confidence score."""
        base_confidence = 0.7
        doc_boost = min(doc_count / 5.0, 0.2)
        
        # Reduce if uncertainty markers present
        uncertainty = 0.1 if any(w in answer.lower() for w in ['might', 'may', 'unclear']) else 0.0
        
        confidence = base_confidence + doc_boost - uncertainty
        return max(0.0, min(1.0, confidence))
    
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
