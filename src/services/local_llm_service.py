import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

class LocalFinetunedService:
    def __init__(self, base_model_name="meta-llama/Meta-Llama-3-8B-Instruct", adapter_path="./models/sg_legal_qa_lora"):
        print("Loading base model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model in 4-bit to save VRAM
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            device_map="auto"
        )
        
        print("Fusing LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1 # Low temp for factual legal answers
        )

    def generate_answer(self, query: str) -> str:
        """Mimics the interface of your existing LLMService"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional Singapore Legal Assistant. Answer the user's question accurately based on Singapore law. You must cite the relevant statutes.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        result = self.pipe(prompt)
        # Strip out the prompt from the response
        generated_text = result[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return generated_text