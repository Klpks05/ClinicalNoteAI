import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "outputs/mts_train.json"
VAL_FILE = "outputs/mts_validation.json"

OUTPUT_DIR = "outputs/qwen_clinical_lora_mts_v2"

MAX_LENGTH = 1024
MAX_TARGET_TOKENS = 300

EPOCHS = 1

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

LEARNING_RATE = 1e-4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print("Training examples:", len(train_data))
print("Validation examples:", len(val_data))


train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a clinical note generation assistant.

Generate ONLY the requested clinical note section.

Use ONLY information explicitly supported by the doctor-patient
conversation.

Do not invent or assume symptoms, history, medications, allergies,
diagnoses, examination findings, tests, treatments, family history,
or social history.

Do not add information from other clinical note sections.

Be concise, factual, and medically accurate.

Output ONLY the requested clinical note text.
Do not provide explanations.
Do not repeat the conversation.
Do not add additional sections."""


# ============================================================
# PREPARE ONE EXAMPLE
# ============================================================

def prepare_example(example):

    section = example["section"]
    dialogue = example["dialogue"]
    target = example["clinical_note"].strip()

    # --------------------------------------------------------
    # Build the exact prompt used during inference
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Generate the clinical note for the section: "
                f"{section}\n\n"
                f"Doctor-patient conversation:\n"
                f"{dialogue}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
    )

    prompt_ids = prompt_tokens["input_ids"]

    # --------------------------------------------------------
    # Tokenize target
    # --------------------------------------------------------

    target_tokens = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TARGET_TOKENS - 1,
    )

    target_ids = target_tokens["input_ids"]

    # --------------------------------------------------------
    # IMPORTANT:
    # Add EOS token so the model learns when to STOP.
    # --------------------------------------------------------

    target_ids = target_ids + [
        tokenizer.eos_token_id
    ]

    # --------------------------------------------------------
    # Keep total length within MAX_LENGTH
    # --------------------------------------------------------

    max_prompt_length = MAX_LENGTH - len(target_ids)

    if max_prompt_length < 1:

        target_ids = target_ids[
            :MAX_TARGET_TOKENS
        ]

        max_prompt_length = MAX_LENGTH - len(target_ids)

    # --------------------------------------------------------
    # Trim long conversations while preserving
    # beginning and end of the conversation.
    # --------------------------------------------------------

    if len(prompt_ids) > max_prompt_length:

        keep_start = max_prompt_length // 2
        keep_end = max_prompt_length - keep_start

        prompt_ids = (
            prompt_ids[:keep_start]
            + prompt_ids[-keep_end:]
        )

    # --------------------------------------------------------
    # Combine prompt + target
    # --------------------------------------------------------

    input_ids = prompt_ids + target_ids

    attention_mask = [1] * len(input_ids)

    # --------------------------------------------------------
    # CRITICAL:
    # Do NOT calculate loss on the prompt.
    #
    # Only calculate loss on the clinical-note target.
    # --------------------------------------------------------

    labels = (
        [-100] * len(prompt_ids)
        + target_ids
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    assert len(input_ids) == len(labels)

    assert any(
        label != -100
        for label in labels
    )

    assert labels[-1] == tokenizer.eos_token_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# TOKENIZE DATASETS
# ============================================================

print("\nPreparing training examples...")

train_dataset = train_dataset.map(
    prepare_example,
    remove_columns=train_dataset.column_names,
)

val_dataset = val_dataset.map(
    prepare_example,
    remove_columns=val_dataset.column_names,
)

print("Training dataset prepared.")
print("Validation dataset prepared.")


# ============================================================
# CHECK EXAMPLES
# ============================================================

print("\nChecking prepared examples...")

for i in range(min(3, len(train_dataset))):

    example = train_dataset[i]

    valid_labels = [
        x for x in example["labels"]
        if x != -100
    ]

    print(
        f"Example {i + 1}: "
        f"sequence={len(example['input_ids'])}, "
        f"target={len(valid_labels)}"
    )

    assert len(valid_labels) > 0


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
)

model = model.to(device)

print("Base model loaded.")


# ============================================================
# LoRA
# ============================================================

print("\nApplying LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

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
# GRADIENT CHECKPOINTING
# ============================================================

model.gradient_checkpointing_enable()

model.config.use_cache = False


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRAD_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    warmup_steps=8,

    weight_decay=0.01,

    logging_steps=20,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    save_total_limit=1,

    report_to="none",

    optim="adamw_torch",

    dataloader_pin_memory=False,

    remove_unused_columns=False,

    use_cpu=False,
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=data_collator,
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING LoRA TRAINING")
print("=" * 70)

trainer.train()


# ============================================================
# SAVE
# ============================================================

print("\n")
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    f"\nModel saved to:\n{OUTPUT_DIR}"
)


# ============================================================
# FINAL EVALUATION LOSS
# ============================================================

print("\nRunning final validation...")

metrics = trainer.evaluate()

print("\nValidation metrics:")

for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


print("\nTraining complete.")