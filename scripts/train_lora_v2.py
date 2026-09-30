import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "outputs/train.json"
VAL_FILE = "outputs/validation.json"

OUTPUT_DIR = "outputs/qwen_clinical_lora_v2"


# ==============================
# DEVICE
# ==============================

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)


# ==============================
# LOAD DATA
# ==============================

print("\nLoading datasets...")

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    train_data = json.load(file)

with open(VAL_FILE, "r", encoding="utf-8") as file:
    val_data = json.load(file)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))


# ==============================
# TOKENIZER
# ==============================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==============================
# PREPARE DATA
# ==============================

def tokenize_example(example):

    conversation = example["input"]
    clinical_note = example["output"]

    prompt = (
        "Convert the following doctor-patient conversation "
        "into a concise clinical note.\n\n"
        "Only include information that is explicitly present "
        "in the conversation. Do not invent clinical information.\n\n"
        "### Conversation:\n"
        + conversation
        + "\n\n"
        "### Clinical Note:\n"
    )

    target = clinical_note + tokenizer.eos_token

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=True
    )

    target_tokens = tokenizer(
        target,
        add_special_tokens=False
    )

    input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]

    attention_mask = [1] * len(input_ids)

    # Ignore prompt tokens during loss calculation
    labels = (
        [-100] * len(prompt_tokens["input_ids"])
        + target_tokens["input_ids"]
    )

    # Limit sequence length
    max_length = 1024

    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names
)

val_dataset = val_dataset.map(
    tokenize_example,
    remove_columns=val_dataset.column_names
)


# ==============================
# LOAD MODEL
# ==============================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model.config.use_cache = False

model = model.to(device)

print("Model loaded successfully!")


# ==============================
# LoRA
# ==============================

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


# ==============================
# TRAINING
# ==============================

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


# ==============================
# CUSTOM COLLATOR
# ==============================

class ClinicalNoteCollator:

    def __call__(self, features):

        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:

            padding_length = (
                max_length
                - len(feature["input_ids"])
            )

            input_ids.append(
                feature["input_ids"]
                + [tokenizer.pad_token_id] * padding_length
            )

            attention_masks.append(
                feature["attention_mask"]
                + [0] * padding_length
            )

            labels.append(
                feature["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long
            )
        }


data_collator = ClinicalNoteCollator()


# ==============================
# TRAINER
# ==============================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    data_collator=data_collator
)


# ==============================
# START TRAINING
# ==============================

print("\n==============================")
print("STARTING V2 LORA TRAINING")
print("==============================\n")

trainer.train()


# ==============================
# SAVE
# ==============================

print("\nSaving V2 LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


print("\n==============================")
print("V2 TRAINING COMPLETE")
print("==============================")

print("Model saved to:")
print(OUTPUT_DIR)