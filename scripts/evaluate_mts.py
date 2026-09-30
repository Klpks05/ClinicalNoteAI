import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "outputs/qwen_clinical_lora_mts"
TEST_FILE = "outputs/mts_test.json"

NUM_EXAMPLES = 10
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 80

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)


# ============================================================
# LOAD DATA
# ============================================================

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_data = test_data[:NUM_EXAMPLES]

print("Examples to evaluate:", len(test_data))


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a clinical note generation assistant.

Generate the requested clinical note section using ONLY information
supported by the doctor-patient conversation.

Do not invent symptoms, history, medications, allergies, diagnoses,
examination findings, tests, or treatments.

Be concise and medically accurate."""


def build_prompt(row):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"Generate the clinical note for the section: "
                f"{row['section']}\n\n"
                f"Doctor-patient conversation:\n"
                f"{row['dialogue']}"
            )
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# ============================================================
# GENERATION
# ============================================================

def generate(model, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = output[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
)

base_model = base_model.to(device)
base_model.eval()

print("Base model loaded.")


# ============================================================
# LOAD LORA
# ============================================================

print("\nLoading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)

model = model.to(device)
model.eval()

print("LoRA model loaded.")


# ============================================================
# ROUGE
# ============================================================

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)


base_predictions = []
lora_predictions = []
references = []


# ============================================================
# EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("STARTING EVALUATION")
print("=" * 70)


for i, row in enumerate(test_data):

    print(
        f"\nProcessing {i + 1}/{len(test_data)}..."
    )

    prompt = build_prompt(row)

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------

    with model.disable_adapter():

        base_prediction = generate(
            model,
            prompt
        )

    # --------------------------------------------------------
    # LORA MODEL
    # --------------------------------------------------------

    lora_prediction = generate(
        model,
        prompt
    )

    base_predictions.append(base_prediction)
    lora_predictions.append(lora_prediction)
    references.append(row["clinical_note"])

    print("\nSECTION:")
    print(row["section"])

    print("\nEXPECTED:")
    print(row["clinical_note"])

    print("\nBASE:")
    print(base_prediction)

    print("\nLORA:")
    print(lora_prediction)

    print("-" * 70)


# ============================================================
# CALCULATE ROUGE
# ============================================================

def calculate_scores(predictions):

    rouge1 = []
    rouge2 = []
    rougeL = []

    for prediction, reference in zip(
        predictions,
        references
    ):

        scores = scorer.score(
            reference,
            prediction
        )

        rouge1.append(
            scores["rouge1"].fmeasure
        )

        rouge2.append(
            scores["rouge2"].fmeasure
        )

        rougeL.append(
            scores["rougeL"].fmeasure
        )

    return {
        "rouge1": sum(rouge1) / len(rouge1),
        "rouge2": sum(rouge2) / len(rouge2),
        "rougeL": sum(rougeL) / len(rougeL)
    }


base_scores = calculate_scores(
    base_predictions
)

lora_scores = calculate_scores(
    lora_predictions
)


# ============================================================
# LENGTH
# ============================================================

def average_words(predictions):

    total = 0

    for prediction in predictions:
        total += len(prediction.split())

    return total / len(predictions)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nBASE QWEN")
print("-" * 40)

print(
    f"ROUGE-1 : {base_scores['rouge1']:.4f}"
)

print(
    f"ROUGE-2 : {base_scores['rouge2']:.4f}"
)

print(
    f"ROUGE-L : {base_scores['rougeL']:.4f}"
)

print(
    f"Average words : "
    f"{average_words(base_predictions):.2f}"
)


print("\nLORA QWEN")
print("-" * 40)

print(
    f"ROUGE-1 : {lora_scores['rouge1']:.4f}"
)

print(
    f"ROUGE-2 : {lora_scores['rouge2']:.4f}"
)

print(
    f"ROUGE-L : {lora_scores['rougeL']:.4f}"
)

print(
    f"Average words : "
    f"{average_words(lora_predictions):.2f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = []

for i, row in enumerate(test_data):

    results.append(
        {
            "id": row["id"],
            "section": row["section"],
            "expected": row["clinical_note"],
            "base_prediction": base_predictions[i],
            "lora_prediction": lora_predictions[i]
        }
    )


output = {
    "num_examples": NUM_EXAMPLES,

    "base_scores": base_scores,

    "lora_scores": lora_scores,

    "base_average_words": average_words(
        base_predictions
    ),

    "lora_average_words": average_words(
        lora_predictions
    ),

    "examples": results
}


with open(
    "outputs/mts_evaluation_10.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nSaved:"
    "\noutputs/mts_evaluation_10.json"
)