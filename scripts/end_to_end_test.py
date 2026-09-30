import torch
import re

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


print("=" * 70)
print("CLINICALNOTEAI - TEXTGRID -> QWEN CLINICAL EXTRACTION")
print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("\nDevice:", device)


# ============================================================
# TEXTGRID FILES
# ============================================================

DOCTOR_TEXTGRID = (
    "data/primock57/transcripts/"
    "day1_consultation01_doctor.TextGrid"
)

PATIENT_TEXTGRID = (
    "data/primock57/transcripts/"
    "day1_consultation01_patient.TextGrid"
)


# ============================================================
# PARSE TEXTGRID
# ============================================================

def parse_textgrid(path, speaker):

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = (
        r"xmin\s*=\s*([0-9.]+)\s*"
        r"xmax\s*=\s*([0-9.]+)\s*"
        r"text\s*=\s*\"(.*?)\""
    )

    matches = re.findall(
        pattern,
        content,
        flags=re.DOTALL
    )

    utterances = []

    for start, end, text in matches:

        text = text.strip()

        if not text:
            continue

        # Remove annotation markers
        text = re.sub(
            r"<UNSURE>(.*?)</UNSURE>",
            r"\1",
            text
        )

        text = text.replace(
            "<UNIN/>",
            ""
        )

        # Clean whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if not text:
            continue

        utterances.append({
            "start": float(start),
            "end": float(end),
            "speaker": speaker,
            "text": text
        })

    return utterances


# ============================================================
# LOAD BOTH SPEAKERS
# ============================================================

print("\nReading TextGrid files...")

doctor = parse_textgrid(
    DOCTOR_TEXTGRID,
    "DOCTOR"
)

patient = parse_textgrid(
    PATIENT_TEXTGRID,
    "PATIENT"
)

utterances = doctor + patient

# Sort chronologically
utterances.sort(
    key=lambda x: x["start"]
)


print("Doctor utterances:", len(doctor))
print("Patient utterances:", len(patient))
print("Total utterances:", len(utterances))


# ============================================================
# BUILD CLEAN CONVERSATION
# ============================================================

conversation_lines = []

for item in utterances:

    conversation_lines.append(
        f"{item['speaker']}: {item['text']}"
    )

conversation = "\n".join(
    conversation_lines
)


# ============================================================
# SHOW CONVERSATION
# ============================================================

print("\n" + "=" * 70)
print("CLEAN DOCTOR-PATIENT CONVERSATION")
print("=" * 70)

print(conversation)


# ============================================================
# LOAD QWEN
# ============================================================

print("\n" + "=" * 70)
print("LOADING QWEN")
print("=" * 70)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
).to(device)

model.eval()

print("Qwen loaded.")


# ============================================================
# CLINICAL EXTRACTION PROMPT
# ============================================================

prompt = f"""
You are a clinical information extraction assistant.

Extract ONLY information explicitly stated in the
doctor-patient conversation.

IMPORTANT RULES:

1. Never invent information.
2. Never infer information.
3. Never change a fact.
4. Preserve left/right exactly.
5. Preserve numbers exactly.
6. Do not add a diagnosis.
7. Do not add examination findings.
8. If information is not present, write "Not mentioned."

Extract these fields:

CHIEF COMPLAINT:
DURATION:
FREQUENCY:
STOOL CHARACTER:
PAIN LOCATION:
PAIN CHARACTER:
PAIN TIMING:
ASSOCIATED SYMPTOMS:
NEGATIVE FINDINGS:

Return ONLY these fields.

Conversation:

{conversation}

Clinical information:
"""


# ============================================================
# TOKENIZE
# ============================================================

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
    max_length=2048
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


print(
    "\nQwen input tokens:",
    inputs["input_ids"].shape[1]
)


# ============================================================
# GENERATE
# ============================================================

print("\nExtracting clinical information...")

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )


input_length = inputs["input_ids"].shape[1]

generated_tokens = output[0][input_length:]

result = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURED CLINICAL INFORMATION")
print("=" * 70)

print(result)


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)