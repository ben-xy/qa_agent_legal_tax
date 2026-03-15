from src.agents.qa_agent import QAAgent
# Import your RAG config and Local Finetuned config...

def run_comparison():
    questions = load_eval_questions("data/qa_pairs/eval_ground_truth.jsonl")
    
    rag_agent = QAAgent(config=rag_config)
    ft_agent = QAAgent(config=finetuned_config)
    
    for q in questions:
        print(f"Question: {q}")
        
        rag_response = rag_agent.process_query(q)
        print(f"[RAG Output]: {rag_response.answer}")
        
        ft_response = ft_agent.process_query(q)
        print(f"[Fine-Tuned Output]: {ft_response.answer}")
        print("-" * 50)

if __name__ == "__main__":
    run_comparison()