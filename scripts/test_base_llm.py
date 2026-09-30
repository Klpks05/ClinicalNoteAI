import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# MODEL CONFIGURATION
# ==========================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)


# ==========================================
# LOAD TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# LOAD BASE MODEL
# ==========================================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
)

model = model.to(device)
model.eval()

print("Base model loaded successfully!")


# ==========================================
# TEST CONVERSATION
# ==========================================

conversation = """Doctor: What brings you in today?

Patient: I've had diarrhea for three days. It's mostly watery and I'm going about six times a day.

Doctor: Any blood in the stool?

Patient: No blood.

Doctor: Any abdominal pain?

Patient: Yes, crampy pain in the lower left side."""


# ==========================================
# SECTION
# ==========================================

section = "CC"


# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """You are a clinical note generation assistant.

Generate the requested clinical note section using ONLY information
supported by the doctor-patient conversation.

Do not invent symptoms, history, medications, allergies, diagnoses,
examination findings, tests, or treatments.

Be concise and medically accurate."""


# ==========================================
# BUILD PROMPT
# ==========================================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": (
            f"Generate the clinical note for the section: {section}\n\n"
            f"Doctor-patient conversation:\n"
            f"{conversation}"
        ),
    },
]


# ==========================================
# CHAT TEMPLATE
# ==========================================

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


# ==========================================
# TOKENIZE
# ==========================================

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1024
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ==========================================
# GENERATE
# ==========================================

print(f"\nGenerating {section} section...\n")

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )


# ==========================================
# EXTRACT OUTPUT
# ==========================================

input_length = inputs["input_ids"].shape[1]

generated_tokens = output[0][input_length:]

result = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()


# ==========================================
# DISPLAY
# ==========================================

print("==============================")
print(f"BASE QWEN OUTPUT — {section}")
print("==============================")
print(result)
print("==============================")