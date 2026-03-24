import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
DATASET_PATH = "data/finetune_dataset/train.jsonl" 
OUTPUT_DIR = "models/sg_legal_qa_lora"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Setup 4-bit Quantization (Crucial to fit the model on Colab's free T4 GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 3. Load Tokenizer and Base Model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 4. Configure LoRA
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# model = get_peft_model(model, peft_config)

# 5. Load and Format the Dataset 
# Assuming your JSONL has 'question' and 'answer' keys
print(f"Loading dataset from {DATASET_PATH}...")
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have placed your training data at the DATASET_PATH.")
    exit()

def formatting_prompts_func(example):
    """Formats a single example for legal question answering."""
    # Note: When using SFTTrainer with a formatting_func, the function 
    # receives ONE example (a dictionary) and should return ONE string.
    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional Singapore Legal Assistant. Answer the user's question accurately based on Singapore law. You must cite the relevant statutes (e.g., Employment Act 1968, s 14).
<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['answer']}<|eot_id|>"""
    return text

# 6. Setup Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8, 
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    max_grad_norm=0.3,
    max_steps=500, # You can increase this depending on how much time you have in Colab
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

# 7. Initialize Trainer with SFTConfig
sft_config = SFTConfig(**training_args.to_dict())
sft_config.max_seq_length = 1024
sft_config.dataset_text_field = "text"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer, # Newer versions use processing_class or handle via model
    args=sft_config,
)

print("Starting training...")
trainer.train()

# 8. Save the Fine-Tuned Adapter
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"QA LoRA adapter successfully saved to {OUTPUT_DIR}")