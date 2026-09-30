import json
import re
from pathlib import Path


# Paths
notes_folder = Path("data/primock57/notes")
transcripts_folder = Path("data/primock57/transcripts")
output_file = Path("outputs/primock57_training_data.json")


def clean_text(text):
    # Remove annotation tags
    text = re.sub(r"<UNSURE>", "", text)
    text = re.sub(r"</UNSURE>", "", text)
    text = re.sub(r"<UNIN/>", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_textgrid(file_path, speaker):

    dialogue = []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    i = 0

    while i < len(lines):

        if lines[i].strip().startswith("xmin ="):

            start_time = float(
                lines[i].split("=")[1].strip()
            )

            i += 1

            if lines[i].strip().startswith("xmax ="):

                end_time = float(
                    lines[i].split("=")[1].strip()
                )

                i += 1

                if lines[i].strip().startswith("text ="):

                    text = lines[i].split("=", 1)[1].strip()

                    text = text.strip('"')
                    text = clean_text(text)

                    if text != "":
                        dialogue.append({
                            "start": start_time,
                            "end": end_time,
                            "speaker": speaker,
                            "text": text
                        })

        i += 1

    return dialogue


# Store all consultations
dataset = []


# Read every clinical note
note_files = sorted(notes_folder.glob("*.json"))


for note_file in note_files:

    # Skip README if present
    if note_file.name.lower() == "readme.json":
        continue

    try:

        # Read clinical note
        with open(note_file, "r", encoding="utf-8") as file:
            note_data = json.load(file)

        day = note_data["day"]
        consultation = note_data["consultation"]

        # Create transcript filenames
        doctor_file = (
            transcripts_folder
            / f"day{day}_consultation{consultation:02d}_doctor.TextGrid"
        )

        patient_file = (
            transcripts_folder
            / f"day{day}_consultation{consultation:02d}_patient.TextGrid"
        )

        # Check files exist
        if not doctor_file.exists():
            print("Missing doctor transcript:", doctor_file)
            continue

        if not patient_file.exists():
            print("Missing patient transcript:", patient_file)
            continue

        # Read transcripts
        doctor_dialogue = read_textgrid(
            doctor_file,
            "DOCTOR"
        )

        patient_dialogue = read_textgrid(
            patient_file,
            "PATIENT"
        )

        # Combine both speakers
        dialogue = doctor_dialogue + patient_dialogue

        # Sort according to conversation time
        dialogue.sort(key=lambda x: x["start"])

        # Create dialogue text
        dialogue_text = ""

        for item in dialogue:

            dialogue_text += (
                f'{item["speaker"]}: '
                f'{item["text"]}\n'
            )

        # Get clinical note
        clinical_note = note_data.get("note", "")

        # Create one training example
        record = {
            "id": f"day{day}_consultation{consultation:02d}",
            "dialogue": dialogue_text.strip(),
            "clinical_note": clinical_note.strip()
        }

        dataset.append(record)

        print(
            f"Processed: day {day}, "
            f"consultation {consultation}"
        )

    except Exception as e:

        print(
            f"Error processing {note_file.name}: {e}"
        )


# Save complete dataset
with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        dataset,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n==============================")
print("DATASET CREATION COMPLETE")
print("==============================")

print("Total consultations:", len(dataset))

print(
    "Saved to:",
    output_file
)