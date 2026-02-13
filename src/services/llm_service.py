"""
LLM service for answer generation using OpenAI GPT API.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based answer generation."""
    
    def __init__(self, config: Dict):
        """
        Initialize LLM service.
        
        Args:
            config: Configuration dictionary with API keys and model names
        """
        self.config = config
        self.model = config.get('llm_model', 'gpt-4-turbo-preview')
        self.temperature = config.get('llm_temperature', 0.3)
        self.max_tokens = config.get('llm_max_tokens', 2000)
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.get('openai_api_key'))
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    def generate_answer(self, 
                       query: str,
                       context: List[Dict],
                       company_info: Optional[Dict] = None,
                       query_type: str = 'general') -> str:
        """
        Generate answer using LLM with context.
        
        Args:
            query: User question
            context: Retrieved documents
            company_info: Optional company context
            query_type: Type of query for prompt selection
        
        Returns:
            Generated answer string
        """
        
        try:
            # Prepare context
            context_text = self._format_context(context, max_length=4000)
            
            # Get system prompt
            system_prompt = self._get_system_prompt(query_type)
            
            # Prepare user message
            user_message = self._prepare_user_message(
                query=query,
                context=context_text,
                company_info=company_info
            )
            
            logger.debug(f"System prompt: {system_prompt[:100]}...")
            logger.debug(f"User prompt length: {len(user_message)} chars")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.95
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.info(f"Generated answer (tokens: {response.usage.completion_tokens})")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _format_context(self, documents: List[Dict], max_length: int = 4000) -> str:
        """Format retrieved documents as context for LLM."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for doc in documents:
            source = doc.get('source', 'Unknown Source')
            content = doc.get('content', '')[:500]
            
            snippet = f"Source: {source}\nContent: {content}\n"
            
            if current_length + len(snippet) > max_length:
                break
            
            context_parts.append(snippet)
            current_length += len(snippet)
        
        if not context_parts:
            return "No relevant content available."
        
        return "\n---\n".join(context_parts)
    
    def _get_system_prompt(self, query_type: str = 'general') -> str:
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
    
    def _prepare_user_message(self, query: str, 
                             context: str,
                             company_info: Optional[Dict] = None) -> str:
        """Prepare user message with context."""
        company_context = ""
        if company_info:
            company_type = company_info.get('company_type', 'Private Limited Company')
            company_context = f"\n\nCompany Type: {company_type}"
        
        return f"""Based on the following documents, answer this question:

{context}

{company_context}

Question: {query}

Please provide a comprehensive, accurate answer with relevant legal citations."""
