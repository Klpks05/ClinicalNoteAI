import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


print("=" * 70)
print("CLINICALNOTEAI - MTS DIALOG -> QWEN BASELINE")
print("=" * 70)

# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("\nDevice:", device)


# --------------------------------------------------
# 2. Load MTS-Dialog
# --------------------------------------------------

print("\nLoading MTS-Dialog...")

dataset = load_dataset("lyumengxian/MTS-Dialog")

print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))
print("Test:", len(dataset["test"]))


# --------------------------------------------------
# 3. Pick one example
# --------------------------------------------------
# We deliberately use a CC example because it is
# relatively simple and easy to inspect.

example = None

for row in dataset["train"]:
    if row["section_header"] == "CC":
        example = row
        break

if example is None:
    raise RuntimeError("Could not find a CC example.")


dialogue = example["dialogue"]
expected = example["section_text"]
section = example["section_header"]


print("\n" + "=" * 70)
print("SELECTED EXAMPLE")
print("=" * 70)

print("\nSection:", section)

print("\n--- CONVERSATION ---")
print(dialogue)

print("\n--- EXPECTED CLINICAL NOTE ---")
print(expected)


# --------------------------------------------------
# 4. Load Qwen
# --------------------------------------------------

print("\n" + "=" * 70)
print("LOADING QWEN")
print("=" * 70)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to(device)

model.eval()

print("Qwen loaded.")


# --------------------------------------------------
# 5. Create prompt
# --------------------------------------------------

prompt = f"""
You are a clinical note generation assistant.

Generate ONLY the requested clinical note section.

Requested section:
{section}

Use ONLY information explicitly stated in the
doctor-patient conversation.

Do NOT:
- invent information
- infer information
- add information from other sections
- add diagnoses unless explicitly stated
- add examination findings unless explicitly stated
- add medications unless explicitly stated

Be concise and factual.

Output ONLY the clinical note text.
Do not explain your answer.

Doctor-patient conversation:

{dialogue}

Clinical note:
"""


# --------------------------------------------------
# 6. Apply Qwen chat template
# --------------------------------------------------

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


# --------------------------------------------------
# 7. Tokenize
# --------------------------------------------------

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

print("\nInput tokens:", inputs["input_ids"].shape[1])


# --------------------------------------------------
# 8. Generate
# --------------------------------------------------

print("\nGenerating...")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )


# --------------------------------------------------
# 9. Decode ONLY generated tokens
# --------------------------------------------------

input_length = inputs["input_ids"].shape[1]

generated_tokens = output[0][input_length:]

generated_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()


# --------------------------------------------------
# 10. Compare
# --------------------------------------------------

print("\n" + "=" * 70)
print("QWEN GENERATED NOTE")
print("=" * 70)

print(generated_text)

print("\n" + "=" * 70)
print("EXPECTED MTS NOTE")
print("=" * 70)

print(expected)

print("\n" + "=" * 70)
print("BASELINE TEST COMPLETE")
print("=" * 70)