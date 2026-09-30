import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "outputs/qwen_clinical_lora_mts"
TEST_FILE = "outputs/mts_test.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

NUM_EXAMPLES = 5
MAX_NEW_TOKENS = 150


# ============================================================
# LOAD DATA
# ============================================================

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("Using device:", DEVICE)
print("Test examples:", len(test_data))


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a clinical note generation assistant.

Generate the requested clinical note section using ONLY information
supported by the doctor-patient conversation.

Do not invent symptoms, history, medications, allergies, diagnoses,
examination findings, tests, or treatments.

Be concise and medically accurate."""


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(example):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Generate the clinical note for the section: "
                f"{example['section']}\n\n"
                f"Doctor-patient conversation:\n"
                f"{example['dialogue']}"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# GENERATE
# ============================================================

def generate(model, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = output[0][input_length:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return result


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
)

base_model = base_model.to(DEVICE)
base_model.eval()

print("Base model loaded.")


# ============================================================
# LOAD LORA MODEL
# ============================================================

print("\nLoading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
)

lora_model = lora_model.to(DEVICE)
lora_model.eval()

print("LoRA model loaded.")


# ============================================================
# EVALUATE
# ============================================================

for i, example in enumerate(test_data[:NUM_EXAMPLES], start=1):

    print("\n")
    print("=" * 75)
    print(f"TEST EXAMPLE {i}")
    print("=" * 75)

    print("\nSECTION:")
    print(example["section"])

    print("\nEXPECTED:")
    print(example["clinical_note"])

    prompt = build_prompt(example)

    # ========================================================
    # TRUE BASE MODEL
    # ========================================================

    print("\n" + "-" * 75)
    print("BASE QWEN")
    print("-" * 75)

    with lora_model.disable_adapter():
        base_result = generate(
            lora_model,
            prompt,
        )

    print(base_result)

    # ========================================================
    # LORA MODEL
    # ========================================================

    print("\n" + "-" * 75)
    print("LORA QWEN")
    print("-" * 75)

    lora_result = generate(
        lora_model,
        prompt,
    )

    print(lora_result)


print("\n")
print("=" * 75)
print("COMPARISON COMPLETE")
print("=" * 75)