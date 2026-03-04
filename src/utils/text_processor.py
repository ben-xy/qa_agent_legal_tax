"""
Text processing utilities.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Utility class for text processing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except punctuation
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_sections(text: str) -> List[Dict[str, str]]:
        """
        Extract legal sections from text.
        
        Args:
            text: Legal document text
        
        Returns:
            List of sections with titles and content
        """
        # Pattern for section headers: "Section 123", "S. 456", etc.
        pattern = r'(?:Section|S\.)\s+(\d+[A-Za-z]*(?:\([a-zA-Z0-9]\))*)\s*[-–]\s*(.+?)(?=(?:Section|S\.)\s+\d+|$)'
        
        matches = re.finditer(pattern, text, re.DOTALL)
        sections = []
        
        for match in matches:
            section_num = match.group(1).strip()
            section_content = match.group(2).strip()
            
            sections.append({
                'section': section_num,
                'title': section_content.split('\n')[0][:100],
                'content': section_content
            })
        
        return sections
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """
        Extract legal citations from text.
        
        Args:
            text: Text containing legal citations
        
        Returns:
            List of citations
        """
        # Pattern for citations: "Section 123(4)(a)", "Act 2020", etc.
        patterns = [
            r'(?:Section|S\.)\s+(\d+[A-Za-z]*(?:\([a-zA-Z0-9]\))*)',
            r'(?:Act|Regulation)\s+(\d{4})',
            r'(?:Chapter)\s+(\d+[A-Z]?)',
            r'(?:Article|Rule)\s+(\d+)'
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            citations.extend([match.group(0) for match in matches])
        
        return list(set(citations))  # Remove duplicates
