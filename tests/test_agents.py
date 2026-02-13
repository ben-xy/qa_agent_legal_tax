"""
Basic test suite for QA Agent system.
"""

import pytest
from unittest.mock import Mock, patch

from src.agents.qa_agent import QAAgent, AgentResponse
from src.validators.legal_validator import LegalValidator


class TestQAAgent:
    """Test cases for QA Agent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {'retrieval_top_k': 5}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        
        assert agent.retriever == mock_retriever
        assert agent.llm_service == mock_llm
        assert agent.validator == mock_validator
        assert agent.config == config
    
    def test_query_classification_tax(self):
        """Test tax query classification."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        
        assert agent._classify_query("What is the GST rate?") == 'tax'
        assert agent._classify_query("How do I file income tax?") == 'tax'
    
    def test_query_classification_financial(self):
        """Test financial query classification."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        
        assert agent._classify_query("How to prepare a balance sheet?") == 'financial'
        assert agent._classify_query("What is income statement?") == 'financial'
    
    def test_query_classification_compliance(self):
        """Test compliance query classification."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        
        assert agent._classify_query("What is the filing deadline?") == 'compliance'
    
    def test_query_classification_general(self):
        """Test general query classification."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        
        assert agent._classify_query("Tell me something?") == 'general'
    
    def test_citation_extraction(self):
        """Test citation extraction."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        
        text = "According to Section 123 of the Income Tax Act 1947, companies must..."
        citations = agent._extract_citations(text)
        
        assert len(citations) > 0
        assert any('Section' in c for c in citations)
    
    def test_conversation_history(self):
        """Test conversation history tracking."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {'retrieval_top_k': 5}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        
        # Add mock responses to history manually
        agent.query_history.append({'query': 'test', 'response': {}})
        
        history = agent.get_conversation_history()
        assert len(history) == 1
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_validator = Mock()
        config = {}
        
        agent = QAAgent(mock_retriever, mock_llm, mock_validator, config)
        agent.query_history = [{'query': 'test'}]
        
        agent.clear_history()
        assert len(agent.query_history) == 0


class TestLegalValidator:
    """Test cases for Legal Validator."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        config = {}
        validator = LegalValidator(config)
        
        assert validator.config == config
    
    def test_validate_answer_short(self):
        """Test validation of short answer."""
        validator = LegalValidator({})
        
        result = validator.validate_answer("short", [], "query")
        
        assert result['is_valid'] == False
        assert result['accuracy_score'] == 1
    
    def test_extract_citations(self):
        """Test citation extraction."""
        validator = LegalValidator({})
        
        text = "Under Section 105 of the Income Tax Act, companies must file..."
        citations = validator.extract_citations(text)
        
        assert len(citations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
