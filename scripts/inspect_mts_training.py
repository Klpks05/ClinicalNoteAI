import json
from pathlib import Path


print("=" * 70)
print("CLINICALNOTEAI - INSPECT MTS TRAINING FORMAT")
print("=" * 70)


FILE = Path("outputs/mts_train.json")

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nTraining examples:", len(data))


# --------------------------------------------------
# Inspect first 3 examples
# --------------------------------------------------

for i, row in enumerate(data[:3]):

    print("\n" + "=" * 70)
    print(f"EXAMPLE {i + 1}")
    print("=" * 70)

    section = row["section"]
    dialogue = row["dialogue"]
    clinical_note = row["clinical_note"]

    print("\nSECTION:")
    print(section)

    print("\n--- INPUT CONVERSATION ---")
    print(dialogue)

    print("\n--- EXPECTED OUTPUT ---")
    print(clinical_note)

    print("\n--- TRAINING TASK ---")
    print("INPUT:")
    print("Doctor-patient conversation")
    print("+")
    print(f"Requested section: {section}")

    print("\nOUTPUT:")
    print(clinical_note)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)