import json
import re


doctor_file = "data/primock57/transcripts/day1_consultation01_doctor.TextGrid"
patient_file = "data/primock57/transcripts/day1_consultation01_patient.TextGrid"
note_file = "data/primock57/notes/day1_consultation01.json"


def clean_text(text):
    text = re.sub(r"<UNSURE>", "", text)
    text = re.sub(r"</UNSURE>", "", text)
    text = re.sub(r"<UNIN/>", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_textgrid(file_path, speaker):
    dialogue = []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    i = 0

    while i < len(lines):

        if lines[i].strip().startswith("xmin ="):
            start_time = float(lines[i].split("=")[1].strip())

            i += 1

            if lines[i].strip().startswith("xmax ="):
                end_time = float(lines[i].split("=")[1].strip())

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


# Read doctor and patient transcripts
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
    dialogue_text += f'{item["speaker"]}: {item["text"]}\n'


# Read clinical note
with open(note_file, "r", encoding="utf-8") as file:
    note_data = json.load(file)


clinical_note = note_data["note"]


# Display training pair
print("\n==============================")
print("INPUT: DOCTOR-PATIENT DIALOGUE")
print("==============================\n")

print(dialogue_text)


print("\n==============================")
print("TARGET: CLINICAL NOTE")
print("==============================\n")

print(clinical_note)


# Save training pair
training_pair = {
    "dialogue": dialogue_text.strip(),
    "clinical_note": clinical_note.strip()
}


with open(
    "outputs/day1_consultation01_training_pair.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        training_pair,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n==============================")
print("Training pair saved successfully!")
print("==============================")
print(
    "outputs/day1_consultation01_training_pair.json"
)