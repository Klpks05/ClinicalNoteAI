import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "outputs/qwen_lora_smoke"
TRAIN_FILE = "outputs/mts_train.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


print("=" * 70)
print("LORA SEEN-EXAMPLE TEST")
print("=" * 70)

print("Device:", DEVICE)


# --------------------------------------------------
# Load the exact first training example
# --------------------------------------------------

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

example = data[0]

section = example["section"]
dialogue = example["dialogue"]
expected = example["clinical_note"]


print("\nTraining example ID:")
print(example["id"])

print("\nSection:")
print(section)

print("\nDialogue:")
print(dialogue)

print("\nExpected clinical note:")
print(expected)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# Load base model
# --------------------------------------------------

print("Loading Qwen...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
).to(DEVICE)


# --------------------------------------------------
# Load LoRA adapter
# --------------------------------------------------

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)

model.eval()

print("LoRA loaded.")


# --------------------------------------------------
# Create prompt
# --------------------------------------------------

system_prompt = """You are a clinical note generation assistant.

Generate ONLY the requested clinical note section.

Use the doctor-patient conversation to generate the requested section.

Do not add unrelated sections.
Do not repeat the conversation.
Be concise and clinically appropriate.

Output ONLY the requested clinical note section.
"""


user_prompt = f"""Requested section: {section}

Doctor-patient conversation:

{dialogue}

Clinical note:"""


messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]


formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    truncation=True,
    max_length=768
)


inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}


# --------------------------------------------------
# Generate
# --------------------------------------------------

print("\nGenerating...")

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )


input_length = inputs["input_ids"].shape[1]

generated_tokens = output[0][input_length:]

generated = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("\n" + "=" * 70)
print("LORA OUTPUT")
print("=" * 70)

print(generated)


print("\n" + "=" * 70)
print("EXPECTED TRAINING TARGET")
print("=" * 70)

print(expected)


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)