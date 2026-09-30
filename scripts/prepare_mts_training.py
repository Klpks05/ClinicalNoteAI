from datasets import load_dataset
import json
from pathlib import Path

# Load MTS-Dialog
dataset = load_dataset("lyumengxian/MTS-Dialog")

# Sections with enough examples for our first model
USE_SECTIONS = {
    "FAM/SOCHX",
    "GENHX",
    "PASTMEDICALHX",
    "CC",
    "PASTSURGICAL",
    "ALLERGY",
    "ROS",
    "MEDICATIONS",
    "ASSESSMENT",
    "EXAM",
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def prepare_split(split_name):
    data = []

    for row in dataset[split_name]:

        section = row["section_header"]

        if section not in USE_SECTIONS:
            continue

        data.append({
            "id": f"{split_name}_{row['ID']}",
            "section": section,
            "dialogue": row["dialogue"],
            "clinical_note": row["section_text"]
        })

    output_file = OUTPUT_DIR / f"mts_{split_name}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"{split_name}: {len(data)} examples")
    print(f"Saved: {output_file}")


# Preserve the official MTS-Dialog splits
prepare_split("train")
prepare_split("validation")
prepare_split("test")