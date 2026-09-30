import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# MODEL CONFIGURATION
# ==========================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
)

model = base_model.to(device)
model.eval()

print("Base Qwen model loaded successfully!")


# ==========================================
# CLINICAL NOTE GENERATOR
# ==========================================

def generate_clinical_note(conversation):

    prompt = f"""You are a clinical documentation assistant.

Convert the doctor-patient conversation into a structured clinical note.

CRITICAL RULES:

1. Use ONLY information explicitly stated in the conversation.
2. NEVER invent or assume information.
3. NEVER add symptoms that were not mentioned.
4. NEVER add medical history that was not mentioned.
5. NEVER add medications that were not mentioned.
6. NEVER add allergies that were not mentioned.
7. NEVER add examination findings that were not mentioned.
8. NEVER add diagnoses unless explicitly stated by the doctor.
9. NEVER add tests or investigations that were not mentioned.
10. NEVER add treatment or plans that were not mentioned.
11. If a section has no information, write "Not mentioned."
12. Keep the note concise.

Use exactly these sections:

CHIEF COMPLAINT
HISTORY OF PRESENT ILLNESS
PAST MEDICAL HISTORY
MEDICATIONS
ALLERGIES
EXAM
ASSESSMENT
PLAN

Doctor-patient conversation:

{conversation}

Clinical note:
"""

    messages = [
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
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = output[0][input_length:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return result


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    conversation = """Doctor: What brings you in today?

Patient: I've had diarrhea for three days. It's mostly watery and I'm going about six times a day.

Doctor: Any blood in the stool?

Patient: No blood.

Doctor: Any abdominal pain?

Patient: Yes, crampy pain in the lower left side."""

    print("\nGenerating clinical note...\n")

    note = generate_clinical_note(conversation)

    print("=" * 60)
    print("CLINICAL NOTE")
    print("=" * 60)
    print(note)
    print("=" * 60)