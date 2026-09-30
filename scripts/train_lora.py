import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "outputs/train.json"
VAL_FILE = "outputs/validation.json"

OUTPUT_DIR = "outputs/qwen_clinical_lora"


# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

print("\nLoading datasets...")

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    train_data = json.load(file)

with open(VAL_FILE, "r", encoding="utf-8") as file:
    val_data = json.load(file)


train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))


# --------------------------------------------------
# 3. Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# 4. Format examples
# --------------------------------------------------

def format_example(example):

    text = (
        "### Instruction:\n"
        + example["instruction"]
        + "\n\n"
        "### Conversation:\n"
        + example["input"]
        + "\n\n"
        "### Clinical Note:\n"
        + example["output"]
        + tokenizer.eos_token
    )

    return {
        "text": text
    }


train_dataset = train_dataset.map(
    format_example
)

val_dataset = val_dataset.map(
    format_example
)


# --------------------------------------------------
# 5. Tokenize
# --------------------------------------------------

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding=False
    )


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=train_dataset.column_names
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=val_dataset.column_names
)


# --------------------------------------------------
# 6. Load model
# --------------------------------------------------

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model.config.use_cache = False

model = model.to(device)

print("Model loaded!")


# --------------------------------------------------
# 7. Configure LoRA
# --------------------------------------------------

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)


model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# --------------------------------------------------
# 8. Training arguments
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    optim="adamw_torch",
    use_cpu=False
)


# --------------------------------------------------
# 9. Data collator
# --------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# --------------------------------------------------
# 10. Trainer
# --------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=data_collator
)


# --------------------------------------------------
# 11. Train
# --------------------------------------------------

print("\n==============================")
print("STARTING LORA TRAINING")
print("==============================\n")

trainer.train()


# --------------------------------------------------
# 12. Save adapter
# --------------------------------------------------

print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print("Model saved to:")
print(OUTPUT_DIR)