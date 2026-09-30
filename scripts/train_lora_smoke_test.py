import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from peft import LoraConfig, get_peft_model


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "outputs/mts_train.json"
VAL_FILE = "outputs/mts_validation.json"

OUTPUT_DIR = "outputs/qwen_lora_smoke_200"

TRAIN_SIZE = 200
VAL_SIZE = 20

MAX_LENGTH = 768
MAX_TARGET_LENGTH = 200


# ============================================================
# DEVICE
# ============================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 70)
print("CLINICALNOTEAI - 200 EXAMPLE LORA SMOKE TEST")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)


train_data = train_data[:TRAIN_SIZE]
val_data = val_data[:VAL_SIZE]


print("\nTraining examples:", len(train_data))
print("Validation examples:", len(val_data))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a clinical note generation assistant.

Generate the requested clinical note section from the doctor-patient conversation.

Use the conversation as the source of information.

Output only the clinical note.
Do not add unrelated sections.
Do not repeat the conversation.
"""


def build_prompt(example):

    section = example["section"]
    dialogue = example["dialogue"]
    target = example["clinical_note"]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Requested section: {section}\n\n"
                f"Doctor-patient conversation:\n\n"
                f"{dialogue}\n\n"
                f"Clinical note:"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt, target


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_example(example):

    prompt, target = build_prompt(example)

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    target_tokens = tokenizer(
        target,
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
        add_special_tokens=False,
    )

    input_ids = (
        prompt_tokens["input_ids"]
        + target_tokens["input_ids"]
        + [tokenizer.eos_token_id]
    )

    attention_mask = [1] * len(input_ids)

    labels = (
        [-100] * len(prompt_tokens["input_ids"])
        + target_tokens["input_ids"]
        + [tokenizer.eos_token_id]
    )

    # Final safety truncation
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("\nTokenizing...")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

train_dataset = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names,
)

val_dataset = val_dataset.map(
    tokenize_example,
    remove_columns=val_dataset.column_names,
)


# ============================================================
# CHECK TARGET LABELS
# ============================================================

valid_labels = 0

for example in train_dataset:

    valid_labels += sum(
        1 for x in example["labels"]
        if x != -100
    )

print("Valid target labels:", valid_labels)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model.to(DEVICE)


# ============================================================
# LORA
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# TRAINING
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=2,

    learning_rate=1e-4,

    logging_steps=10,

    eval_strategy="steps",
    eval_steps=50,

    save_strategy="no",

    report_to="none",

    fp16=False,
    bf16=False,

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    ),
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

trainer.train()


# ============================================================
# SAVE
# ============================================================

print("\nSaving LoRA adapter...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


print("\n" + "=" * 70)
print("200 EXAMPLE SMOKE TEST COMPLETE")
print("=" * 70)

print("Saved to:", OUTPUT_DIR)