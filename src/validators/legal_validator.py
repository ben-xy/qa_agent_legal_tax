"""
Legal validator for answer validation and citation extraction.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class LegalValidator:
    """Validator for legal answers and citations."""
    
    def __init__(self, config: Dict):
        """Initialize validator with configuration."""
        self.config = config
    
    def validate_answer(self, answer: str, 
                       context: List[Dict],
                       query: str) -> Dict:
        """
        Validate answer quality.
        
        Args:
            answer: Generated answer
            context: Retrieved context documents
            query: Original query
        
        Returns:
            Validation result dictionary
        """
        result = {
            'is_valid': True,
            'accuracy_score': 3,
            'relevance_score': 3,
            'issues': [],
            'confidence': 0.8
        }
        
        # Basic validation checks
        if not answer or len(answer) < 50:
            result['is_valid'] = False
            result['accuracy_score'] = 1
            result['issues'].append("Answer too short")
        
        if not context:
            result['confidence'] = 0.5
            result['issues'].append("Limited supporting context")
        
        logger.info(f"Answer validated: {result['is_valid']}")
        
        return result
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text."""
        import re
        
        patterns = [
            r'(?:Section|S\.)\s+(\d+[A-Za-z]*(?:\([a-zA-Z0-9]\))*)',
            r'(?:Act)\s+(\d{4})',
            r'(?:Chapter)\s+(\d+)',
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            citations.extend([match.group(0) for match in matches])
        
        return list(set(citations))
