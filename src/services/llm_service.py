"""
LLM service for answer generation using OpenAI GPT API.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based answer generation using OpenAI or Gemini."""
    
    def __init__(self, config: Dict):
        """
        Initialize LLM service.
        
        Args:
            config: Configuration dictionary with API keys and model names
        """
        self.config = config
        self.provider = self._config_get('llm_provider', 'gemini').lower()
        self.model = self._config_get('llm_model', 'gemini-2.5-pro')
        self.temperature = float(self._config_get('llm_temperature', 0.3))
        self.max_tokens = int(self._config_get('llm_max_tokens', 2000))
        self.client = None

        self._init_client()

    def _config_get(self, key: str, default: Any = None) -> Any:
        """Get config value with lowercase + uppercase compatibility."""
        return self.config.get(key, self.config.get(key.upper(), default))

    def _is_retriable_error(self, exc: Exception) -> bool:
        """Return True for transient network/provider errors worth retrying."""
        error_text = str(exc).lower()
        retriable_markers = [
            "connecterror",
            "connection error",
            "timed out",
            "timeout",
            "temporary failure",
            "service unavailable",
            "eof occurred in violation of protocol",
            "ssl",
        ]
        return any(marker in error_text for marker in retriable_markers)

    def _run_with_retry(self, fn, operation_name: str) -> str:
        """Run provider call with retry for transient failures."""
        max_attempts = int(self._config_get("network_retry_attempts", 3))
        base_delay = float(self._config_get("network_retry_delay", 1.0))

        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                should_retry = self._is_retriable_error(exc) and attempt < max_attempts
                if not should_retry:
                    raise

                delay = base_delay * attempt
                logger.warning(
                    "%s failed on attempt %s/%s: %s. Retrying in %.1fs...",
                    operation_name,
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        if last_exc:
            raise last_exc
        raise RuntimeError(f"{operation_name} failed unexpectedly")

    def _init_client(self) -> None:
        """Initialize provider-specific LLM client."""
        if self.provider == 'openai':
            try:
                import openai

                api_key = self._config_get('openai_api_key')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

                self.model = self._config_get('openai_llm_model', self.model)
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError as exc:
                logger.error("OpenAI package not installed")
                raise ImportError("Please install openai package") from exc

        elif self.provider == 'gemini':
            try:
                from google import genai

                api_key = self._config_get('google_api_key')
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")

                self.model = self._config_get('gemini_llm_model', self.model)
                self.client = genai.Client(api_key=api_key)
            except ImportError as exc:
                logger.error("google-genai package not installed")
                raise ImportError("Please install google-genai package") from exc
        else:
            raise ValueError(f"Unsupported llm provider: {self.provider}")
    
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

            if self.provider == 'openai':
                answer = self._run_with_retry(
                    lambda: self._generate_openai(system_prompt, user_message),
                    "OpenAI answer generation"
                )
            else:
                answer = self._run_with_retry(
                    lambda: self._generate_gemini(system_prompt, user_message),
                    "Gemini answer generation"
                )

            logger.info(f"Generated answer using provider={self.provider}, model={self.model}")
            
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

    def _generate_openai(self, system_prompt: str, user_message: str) -> str:
        """Generate answer using OpenAI Chat Completions API."""
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
        return response.choices[0].message.content.strip()

    def _generate_gemini(self, system_prompt: str, user_message: str) -> str:
        """Generate answer using Gemini API."""
        full_prompt = f"{system_prompt}\n\n{user_message}"
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )
        return (response.text or "").strip()
