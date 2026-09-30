import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# Check device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# Load tokenizer
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Load model
print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

model = model.to(device)

print("Model loaded successfully!")


# Test prompt
prompt = """Convert the following doctor-patient conversation into a concise clinical note.

DOCTOR: What brings you in today?
PATIENT: I've had diarrhea for three days. It's mostly watery and I'm going about six times a day.
DOCTOR: Any blood in the stool?
PATIENT: No blood.
DOCTOR: Any abdominal pain?
PATIENT: Yes, crampy pain in the lower left side.

Clinical Note:
"""


# Tokenize
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# Generate
print("\nGenerating clinical note...\n")

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )


# Decode
result = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


print("==============================")
print("MODEL OUTPUT")
print("==============================")
print(result)