import json
from pathlib import Path


# Input and output files
input_file = Path("outputs/primock57_training_data.json")
output_file = Path("outputs/primock57_instruction_data.json")


# Load dataset
with open(input_file, "r", encoding="utf-8") as file:
    dataset = json.load(file)


instruction_data = []


for item in dataset:

    dialogue = item["dialogue"]
    clinical_note = item["clinical_note"]

    # Create instruction
    instruction = (
        "Convert the following doctor-patient conversation "
        "into a concise clinical note. "
        "Include the medically relevant information "
        "and preserve important clinical details."
    )

    # Create instruction-tuning example
    record = {
        "id": item["id"],
        "instruction": instruction,
        "input": dialogue,
        "output": clinical_note
    }

    instruction_data.append(record)


# Save instruction dataset
with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        instruction_data,
        file,
        indent=4,
        ensure_ascii=False
    )


print("==============================")
print("INSTRUCTION DATASET CREATED")
print("==============================")
print("Total records:", len(instruction_data))
print("Saved to:", output_file)