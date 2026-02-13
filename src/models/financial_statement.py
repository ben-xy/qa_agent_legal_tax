"""
Data models for financial statements.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from decimal import Decimal
from datetime import datetime


@dataclass
class BalanceSheet:
    """Balance sheet model."""
    company_name: str
    year_end: str
    assets: Dict
    liabilities: Dict
    equity: Dict
    current_assets: Decimal = 0
    non_current_assets: Decimal = 0
    total_assets: Decimal = 0
    current_liabilities: Decimal = 0
    non_current_liabilities: Decimal = 0
    total_liabilities: Decimal = 0
    total_equity: Decimal = 0
    is_balanced: bool = False


@dataclass
class IncomeStatement:
    """Income statement model."""
    company_name: str
    year_end: str
    revenue: Decimal = 0
    cost_of_sales: Decimal = 0
    gross_profit: Decimal = 0
    operating_expenses: Dict = None
    total_operating_expenses: Decimal = 0
    operating_profit: Decimal = 0
    finance_costs: Decimal = 0
    profit_before_tax: Decimal = 0
    tax_expense: Decimal = 0
    net_profit: Decimal = 0
    
    def __post_init__(self):
        if self.operating_expenses is None:
            self.operating_expenses = {}


@dataclass
class CashFlowStatement:
    """Cash flow statement model."""
    company_name: str
    year_end: str
    operating_activities: Dict = None
    operating_cf_total: Decimal = 0
    investing_activities: Dict = None
    investing_cf_total: Decimal = 0
    financing_activities: Dict = None
    financing_cf_total: Decimal = 0
    net_change_in_cash: Decimal = 0
    opening_cash: Decimal = 0
    closing_cash: Decimal = 0
    
    def __post_init__(self):
        if self.operating_activities is None:
            self.operating_activities = {}
        if self.investing_activities is None:
            self.investing_activities = {}
        if self.financing_activities is None:
            self.financing_activities = {}


@dataclass
class FinancialStatements:
    """Complete set of financial statements."""
    balance_sheet: BalanceSheet
    income_statement: IncomeStatement
    cash_flow_statement: CashFlowStatement
    generated_date: datetime = None
    generated_by: str = "System"
    
    def __post_init__(self):
        if self.generated_date is None:
            self.generated_date = datetime.now()
