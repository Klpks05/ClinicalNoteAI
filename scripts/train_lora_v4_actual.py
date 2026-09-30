import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = Path("outputs/train.json")
VALIDATION_FILE = Path("outputs/validation.json")

OUTPUT_DIR = Path("outputs/qwen_clinical_lora_v4_mps")

MAX_LENGTH = 2048
MAX_TARGET_TOKENS = 600

# For conversations longer than the available space,
# preserve BOTH the beginning and the end.
MAX_CONVERSATION_TOKENS = MAX_LENGTH - MAX_TARGET_TOKENS

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("=" * 70)
print("QWEN CLINICAL NOTE - V4 TRAINING")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Base model: {BASE_MODEL}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Conversation token budget: {MAX_CONVERSATION_TOKENS}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
    validation_data = json.load(f)

print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(validation_data)}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ============================================================
# BUILD TRAINING TEXT
# ============================================================

def build_prompt(instruction, conversation):
    """
    Creates the user prompt used during training.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical note generation assistant. "
                "Generate a concise clinical note using only information "
                "supported by the doctor-patient conversation. "
                "Do not invent symptoms, history, medications, allergies, "
                "diagnoses, examinations, tests, or treatments."
            ),
        },
        {
            "role": "user",
            "content": (
                instruction
                + "\n\n"
                + "DOCTOR-PATIENT CONVERSATION:\n"
                + conversation
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# LONG CONVERSATION HANDLING
# ============================================================

def prepare_conversation_tokens(conversation):
    """
    Tokenize a conversation.

    If it is too long, preserve the beginning and the end.
    """

    tokens = tokenizer(
        conversation,
        add_special_tokens=False,
    ).input_ids

    if len(tokens) <= MAX_CONVERSATION_TOKENS:
        return tokens

    # Preserve both beginning and end.
    head_size = MAX_CONVERSATION_TOKENS // 2
    tail_size = MAX_CONVERSATION_TOKENS - head_size

    head = tokens[:head_size]
    tail = tokens[-tail_size:]

    # Use EOS as a separator between the two pieces.
    separator = [tokenizer.eos_token_id]

    return head + separator + tail


# ============================================================
# TOKENIZATION WITH LABEL MASKING
# ============================================================

def tokenize_example(example):

    instruction = example["instruction"]
    conversation = example["input"]
    target = example["output"]

    # --------------------------------------------------------
    # Prepare conversation
    # --------------------------------------------------------

    conversation_tokens = prepare_conversation_tokens(conversation)

    # --------------------------------------------------------
    # Build the prompt manually so we know exactly which
    # tokens belong to the prompt and which belong to target.
    # --------------------------------------------------------

    prompt_text = build_prompt(
        instruction,
        tokenizer.decode(
            conversation_tokens,
            skip_special_tokens=True,
        ),
    )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
    ).input_ids

    target_tokens = tokenizer(
        target,
        add_special_tokens=False,
    ).input_ids

    # Add EOS after the clinical note.
    target_tokens = target_tokens + [tokenizer.eos_token_id]

    # --------------------------------------------------------
    # Protect target from being truncated
    # --------------------------------------------------------

    if len(target_tokens) > MAX_TARGET_TOKENS:

        target_tokens = target_tokens[:MAX_TARGET_TOKENS]

    available_prompt_tokens = (
        MAX_LENGTH - len(target_tokens)
    )

    # --------------------------------------------------------
    # If prompt is too long, keep its beginning and end.
    # --------------------------------------------------------

    if len(prompt_tokens) > available_prompt_tokens:

        head_size = available_prompt_tokens // 2
        tail_size = available_prompt_tokens - head_size

        prompt_tokens = (
            prompt_tokens[:head_size]
            + [tokenizer.eos_token_id]
            + prompt_tokens[-tail_size:]
        )

        # Safety correction if separator caused +1 token.
        prompt_tokens = prompt_tokens[:available_prompt_tokens]

    # --------------------------------------------------------
    # Combine prompt + target
    # --------------------------------------------------------

    input_ids = prompt_tokens + target_tokens

    # Prompt tokens do NOT contribute to loss.
    labels = (
        [-100] * len(prompt_tokens)
        + target_tokens
    )

    # --------------------------------------------------------
    # Final safety checks
    # --------------------------------------------------------

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            f"Sequence too long: {len(input_ids)}"
        )

    valid_labels = sum(
        1 for x in labels if x != -100
    )

    if valid_labels == 0:
        raise ValueError(
            "No valid target labels found."
        )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# DATASET CREATION
# ============================================================

print("\nPreparing training dataset...")

train_dataset = Dataset.from_list(train_data)
validation_dataset = Dataset.from_list(validation_data)

train_dataset = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names,
)

validation_dataset = validation_dataset.map(
    tokenize_example,
    remove_columns=validation_dataset.column_names,
)

print("Dataset tokenization complete.")


# ============================================================
# LABEL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL VALIDATION")
print("=" * 70)


def check_labels(dataset, name):

    valid_examples = 0
    total_labels = 0

    for example in dataset:

        valid_labels = sum(
            1 for x in example["labels"]
            if x != -100
        )

        if valid_labels > 0:
            valid_examples += 1

        total_labels += valid_labels

    print(
        f"{name}: "
        f"{valid_examples}/{len(dataset)} examples "
        f"have valid labels"
    )

    print(
        f"{name}: total valid target labels = "
        f"{total_labels}"
    )

    if valid_examples != len(dataset):
        raise ValueError(
            f"{name} contains examples with no valid labels."
        )


check_labels(train_dataset, "Training")
check_labels(validation_dataset, "Validation")


# ============================================================
# SHOW ONE TRAINING EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING EXAMPLE CHECK")
print("=" * 70)

example = train_dataset[0]

print(f"Input tokens : {len(example['input_ids'])}")

valid_label_count = sum(
    1 for x in example["labels"]
    if x != -100
)

prompt_label_count = sum(
    1 for x in example["labels"]
    if x == -100
)

print(f"Prompt tokens: {prompt_label_count}")
print(f"Target tokens: {valid_label_count}")

if valid_label_count == 0:
    raise ValueError(
        "First training example has no target labels."
    )

print("\nLabel validation PASSED.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

print("\nLoading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
)

print("Base model loaded.")


# ============================================================
# PREPARE MODEL FOR MPS
# ============================================================

model.config.use_cache = False

if DEVICE == "mps":
    model = model.to("mps")


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    logging_steps=1,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=False,

    save_total_limit=2,

    report_to="none",

    remove_unused_columns=False,

    fp16=False,
    bf16=False,

    optim="adamw_torch",

    gradient_checkpointing=True,

    
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    

    data_collator=data_collator,
)


# ============================================================
# FINAL CONFIRMATION
# ============================================================

print("\n" + "=" * 70)
print("READY TO TRAIN")
print("=" * 70)

print(f"Training examples : {len(train_dataset)}")
print(f"Validation examples: {len(validation_dataset)}")
print(f"Epochs             : {NUM_EPOCHS}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning rate      : {LEARNING_RATE}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Output directory   : {OUTPUT_DIR}")

print("\nStarting training...\n")


# ============================================================
# TRAIN
# ============================================================

trainer.train()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING V4 MODEL")
print("=" * 70)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\nV4 training completed successfully.")

print(f"Model saved to:")
print(OUTPUT_DIR)