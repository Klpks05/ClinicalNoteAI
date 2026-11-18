import sys
from pathlib import Path
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@pytest.fixture
def sample_wav_path():
    path = REPO_ROOT / "data" / "primock57" / "audio" / "day1_consultation01_doctor.wav"
    return path

@pytest.fixture
def doctor_textgrid_path():
    return REPO_ROOT / "data" / "primock57" / "transcripts" / "day1_consultation01_doctor.TextGrid"

@pytest.fixture
def patient_textgrid_path():
    return REPO_ROOT / "data" / "primock57" / "transcripts" / "day1_consultation01_patient.TextGrid"

@pytest.fixture
def sample_conversation_text():
    return (
        "DOCTOR: Good morning, what brings you in today?\n"
        "PATIENT: I've had really watery diarrhea for three days, about 6 times a day, with lower left crampy pain.\n"
        "DOCTOR: Any blood in the stool or fever?\n"
        "PATIENT: No blood. I had a slight fever on the first day. My wife also vomited after Chinese takeaway.\n"
        "DOCTOR: Any medical history or regular medications?\n"
        "PATIENT: Just asthma, I use an inhaler.\n"
        "DOCTOR: I suspect acute gastroenteritis. Rest, drink fluids, take paracetamol if feverish."
    )
