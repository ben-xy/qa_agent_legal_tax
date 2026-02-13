"""
Prompt templates for QA system.
"""

from typing import Optional, Dict


def get_qa_system_prompt(query_type: str = 'general') -> str:
    """Get system prompt based on query type."""
    
    base_prompt = """You are an expert legal and tax advisor for Singapore companies. 
You provide accurate, detailed answers to questions about Singapore law, regulations, and tax matters.

Your responses should:
1. Be accurate and cite relevant laws and regulations
2. Provide practical examples when helpful
3. Mention any important exceptions or conditions
4. Advise when professional consultation is recommended
5. Use clear, professional language suitable for business owners and accountants"""
    
    if query_type == 'tax':
        return base_prompt + """\n\nFor tax-related questions:
- Reference the Income Tax Act, Goods and Services Tax Act, and IRAS guidelines
- Include specific tax treatment and relief provisions
- Mention filing deadlines and penalties for non-compliance"""
    
    elif query_type == 'financial':
        return base_prompt + """\n\nFor financial statement questions:
- Reference Singapore Financial Reporting Standards (SFRS)
- Provide guidance on proper accounting treatment
- Explain required disclosures"""
    
    elif query_type == 'compliance':
        return base_prompt + """\n\nFor compliance questions:
- Clearly state regulatory requirements and deadlines
- Specify penalties and consequences for non-compliance
- Provide practical steps for compliance"""
    
    return base_prompt


def get_qa_user_prompt(query: str, 
                       context: str,
                       company_info: Optional[Dict] = None,
                       query_type: str = 'general') -> str:
    """Get user prompt with context."""
    
    company_context = ""
    if company_info:
        company_type = company_info.get('company_type', 'Private Limited Company')
        company_context = f"\n\nCompany Type: {company_type}"
    
    return f"""Based on the following documents, answer this question:

{context}

{company_context}

Question: {query}

Please provide a comprehensive, accurate answer with relevant legal citations."""
