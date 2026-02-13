"""
Legal regulation model.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Regulation:
    """Model for legal regulations."""
    name: str
    section: str
    title: str
    content: str
    source_url: Optional[str] = None
    effective_date: Optional[str] = None
    related_sections: List[str] = None
    
    def __post_init__(self):
        if self.related_sections is None:
            self.related_sections = []
