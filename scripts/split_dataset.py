import json
import random
from pathlib import Path


input_file = Path("outputs/primock57_instruction_data.json")

train_file = Path("outputs/train.json")
validation_file = Path("outputs/validation.json")


# Load dataset
with open(input_file, "r", encoding="utf-8") as file:
    dataset = json.load(file)


# Shuffle dataset
random.seed(42)
random.shuffle(dataset)


# Calculate split
split_index = int(len(dataset) * 0.8)

train_data = dataset[:split_index]
validation_data = dataset[split_index:]


# Save training data
with open(train_file, "w", encoding="utf-8") as file:
    json.dump(
        train_data,
        file,
        indent=4,
        ensure_ascii=False
    )


# Save validation data
with open(validation_file, "w", encoding="utf-8") as file:
    json.dump(
        validation_data,
        file,
        indent=4,
        ensure_ascii=False
    )


print("==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")

print("Total records:", len(dataset))
print("Training records:", len(train_data))
print("Validation records:", len(validation_data))

print("\nTraining file:", train_file)
print("Validation file:", validation_file)