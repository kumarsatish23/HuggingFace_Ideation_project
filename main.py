import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, default_data_collator
from peft import get_peft_model, LoraConfig, TaskType

# Check for MPS device (Metal Performance Shaders)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Create project directory
project_dir = "./math_model_project"
os.makedirs(project_dir, exist_ok=True)

# Load the MATH-Hard dataset
dataset = load_dataset("lighteval/MATH-Hard")
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Load GPT-2 tokenizer and set padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# Tokenize dataset
def tokenize_function(examples):
    inputs = [f"Problem: {problem} Solution:" for problem in examples['problem']]
    targets = examples['solution']

    # Set max_length for tokenization
    tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128, return_tensors='pt')

    labels['input_ids'][labels['input_ids'] == tokenizer.pad_token_id] = -100

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels['input_ids']
    }


# Apply tokenization
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Convert to PyTorch tensor format
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader with increased num_workers for parallel loading
train_dataloader = DataLoader(tokenized_train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=default_data_collator)
eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=8, num_workers=4, pin_memory=True,
                             collate_fn=default_data_collator)

# Load GPT-2 model and move to device
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

# Training arguments with mixed precision and gradient accumulation
training_args = TrainingArguments(
    output_dir=os.path.join(project_dir, "gpt2-lora-output"),
    per_device_train_batch_size=8,  # Increased batch size
    gradient_accumulation_steps=2,  # Adjust as needed
    num_train_epochs=3,
    logging_dir=os.path.join(project_dir, "logs"),
    logging_steps=100,
    save_steps=500,
    eval_strategy="steps",
    save_total_limit=2,
    learning_rate=5e-5,
    bf16=True,  # Enable mixed precision training
    dataloader_pin_memory=True,  # Enable pin_memory
)

# Initialize Trainer with DataLoader
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=default_data_collator,
)

# Train the model
trainer.train()

# Save model and tokenizer
trainer.save_model(os.path.join(project_dir, "gpt2-lora-finetuned"))
tokenizer.save_pretrained(os.path.join(project_dir, "gpt2-lora-finetuned", "tokenizer"))


# Text generation function
def generate_text(prompt, max_length=100, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=2.0
        )
    return [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]


# Test the model
sample_prompt = "Problem: Solve for x in the equation: 2x + 5 = 15. Solution:"
generated_texts = generate_text(sample_prompt, max_length=100, num_return_sequences=3)

# Output results
for i, text in enumerate(generated_texts):
    print(f"Generated Solution {i + 1}:\n{text}\n")
