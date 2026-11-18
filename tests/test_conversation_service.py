import pytest
from pathlib import Path

from backend.app.services.conversation_service import (
    ConversationService,
    ConversationServiceError,
    Utterance,
)

@pytest.fixture
def service():
    return ConversationService()

def test_clean_text(service):
    raw = "Hello <UNSURE>how are you</UNSURE> <UNIN/> today <INAUDIBLE_SPEECH/>"
    cleaned = service.clean_text(raw)
    assert cleaned == "Hello how are you today"

def test_reconstruct_from_single_transcript_with_labels(service):
    text = "DOCTOR: Good morning. How are you?\nPATIENT: I have a bad cough.\nDOCTOR: For how long?"
    conv = service.reconstruct_from_single_transcript(text)
    assert len(conv.utterances) == 3
    assert conv.utterances[0].speaker == "Doctor"
    assert conv.utterances[0].text == "Good morning. How are you?"
    assert conv.utterances[1].speaker == "Patient"
    assert conv.utterances[1].text == "I have a bad cough."
    assert conv.doctor_utterances_count if hasattr(conv, 'doctor_utterances_count') else len(conv.doctor_utterances) == 2

def test_reconstruct_from_dual_transcripts_with_timestamps(service):
    doc_segments = [
        {"text": "Hello", "start": 0.0, "end": 1.0},
        {"text": "How can I help you?", "start": 2.5, "end": 4.0},
    ]
    pat_segments = [
        {"text": "Hi Doctor", "start": 1.2, "end": 2.2},
        {"text": "I feel dizzy", "start": 4.2, "end": 5.5},
    ]
    conv = service.reconstruct_from_dual_transcripts(
        doctor_text="Hello How can I help you?",
        patient_text="Hi Doctor I feel dizzy",
        doctor_segments=doc_segments,
        patient_segments=pat_segments,
    )
    assert len(conv.utterances) == 4
    # Chronological order: Doc (0.0) -> Pat (1.2) -> Doc (2.5) -> Pat (4.2)
    assert conv.utterances[0].speaker == "Doctor"
    assert conv.utterances[1].speaker == "Patient"
    assert conv.utterances[2].speaker == "Doctor"
    assert conv.utterances[3].speaker == "Patient"

def test_reconstruct_from_primock_textgrids(service, doctor_textgrid_path, patient_textgrid_path):
    if not doctor_textgrid_path.exists() or not patient_textgrid_path.exists():
        pytest.skip("PriMock57 TextGrid files not found")

    conv = service.reconstruct_from_textgrids(
        doctor_textgrid=doctor_textgrid_path,
        patient_textgrid=patient_textgrid_path,
        consultation_id="day1_consultation01",
    )
    assert len(conv.utterances) > 50
    assert len(conv.doctor_utterances) > 20
    assert len(conv.patient_utterances) > 20
    formatted = conv.to_formatted_text()
    assert "DOCTOR:" in formatted
    assert "PATIENT:" in formatted
