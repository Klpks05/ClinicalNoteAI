import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


print("=" * 70)
print("CLINICALNOTEAI - BASE QWEN vs SMOKE LORA")
print("=" * 70)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "outputs/qwen_lora_smoke"

print("\nDevice:", DEVICE)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# Load base model
# --------------------------------------------------

print("\nLoading base Qwen...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
).to(DEVICE)

model.eval()


# --------------------------------------------------
# Load LoRA adapter
# --------------------------------------------------

print("Loading smoke-test LoRA...")

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL
)

model.eval()

print("LoRA loaded.")


# --------------------------------------------------
# Test example
# --------------------------------------------------

section = "CC"

dialogue = """Doctor: Hi, how are you?
Patient: I burned my hand.
Doctor: Oh, I am sorry. Wow!
Patient: Yeah.
Doctor: Is it only right arm?
Patient: Yes."""


expected = "Burn, right arm."


# --------------------------------------------------
# Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """You are a clinical note generation assistant.

Generate ONLY the requested clinical note section.

Use the doctor-patient conversation to generate the requested section.

Do not add unrelated sections.
Do not repeat the conversation.
Be concise and clinically appropriate.

Output ONLY the requested clinical note section.
"""


prompt = f"""Requested section: {section}

Doctor-patient conversation:

{dialogue}

Clinical note:"""


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": prompt
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
# Generation helper
# --------------------------------------------------

def generate():

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = output[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


# --------------------------------------------------
# Base model
# --------------------------------------------------

print("\n" + "=" * 70)
print("BASE QWEN")
print("=" * 70)

with model.disable_adapter():

    base_output = generate()

print(base_output)


# --------------------------------------------------
# LoRA model
# --------------------------------------------------

print("\n" + "=" * 70)
print("SMOKE-TEST LORA")
print("=" * 70)

lora_output = generate()

print(lora_output)


# --------------------------------------------------
# Expected
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXPECTED MTS OUTPUT")
print("=" * 70)

print(expected)


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)