"""
Company information model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Company:
    """Model for company information."""
    name: str
    registration_number: str
    company_type: str  # Private Limited, Public Limited, Sole Proprietor, Partnership
    financial_year_end: str
    industry: Optional[str] = None
    annual_revenue: Optional[float] = None
    number_of_employees: Optional[int] = None
    address: Optional[str] = None
