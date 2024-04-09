import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Set device to CPU
device = torch.device("cpu")

# Assuming you've loaded your dataset as before
dataset = load_dataset("json", data_files="training_data.jsonl")

# Load tokenizer and model (replace "Llama2" with the actual model identifier)
model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")  # Ensure model is on CPU

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments with smaller batch size suitable for CPU
training_args = TrainingArguments(
    output_dir="./llama2_finetuned",
    num_train_epochs=1,  # You might need to reduce epochs or use a smaller dataset due to slower training on CPU
    per_device_train_batch_size=1,  # Reduced batch size for CPU
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    no_cuda=True,  # Ensure that CUDA (GPU) is not used
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Start fine-tuning
trainer.train()
model.save_pretrained("./my_fine_tuned_model")
tokenizer.save_pretrained("./my_fine_tuned_model")

