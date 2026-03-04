"""
QA pair data model.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class QAPair:
    """QA pair model for question-answer pairs."""
    question: str
    answer: str
    source: str
    section: Optional[str] = None
    category: Optional[str] = None  # definition, compliance, procedure, consequence, scenario, general
    confidence_score: float = 1.0
    created_date: datetime = None
    validated: bool = False
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
