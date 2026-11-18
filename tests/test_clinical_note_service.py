import pytest
from unittest.mock import MagicMock

from backend.app.services.clinical_note_service import (
    ClinicalNoteService,
    ClinicalNote,
    NoteMetadata,
    SECTION_MAP,
)
from backend.app.services.model_service import ModelService

@pytest.fixture
def mock_model_service():
    service = MagicMock(spec=ModelService)
    service.is_loaded = True
    service.model_info = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_path": "outputs/qwen_clinical_lora_mts",
    }
    service.build_chat_prompt.return_value = "mock_prompt"
    service.generate.return_value = "Simulated clinical findings: 3 days of diarrhea, no blood."
    return service

@pytest.fixture
def note_service(mock_model_service):
    return ClinicalNoteService(mock_model_service)

def test_truncate_conversation_preserves_head_and_tail(note_service):
    long_conversation = "START_SYMPTOMS " * 200 + "MIDDLE_CHAT " * 500 + "END_PLAN " * 200
    truncated = note_service._truncate_conversation(long_conversation, max_chars=500)
    assert len(truncated) <= 600
    assert "START_SYMPTOMS" in truncated
    assert "END_PLAN" in truncated
    assert "[... conversation truncated for length ...]" in truncated

def test_generate_note_structure(note_service, sample_conversation_text):
    note = note_service.generate_note(sample_conversation_text)
    assert isinstance(note, ClinicalNote)
    assert note.chief_complaint != ""
    assert note.history_of_present_illness != ""
    assert note.past_medical_history != ""
    assert note.medications != ""
    assert note.allergies != ""
    assert note.review_of_systems != ""
    assert note.examination != ""
    assert note.assessment != ""
    assert note.plan != ""
    assert note.metadata.is_draft is True
    assert "DRAFT" in note.metadata.disclaimer

def test_note_to_text(note_service):
    note = ClinicalNote(
        chief_complaint="Watery diarrhea x 3 days",
        assessment="Acute gastroenteritis",
        plan="Push fluids, paracetamol",
    )
    text = note.to_text()
    assert "CHIEF COMPLAINT" in text
    assert "Watery diarrhea x 3 days" in text
    assert "ASSESSMENT" in text
    assert "Acute gastroenteritis" in text
    assert "PLAN" in text
    assert "DRAFT" in text

def test_validate_note_valid(note_service):
    note = ClinicalNote(chief_complaint="Cough")
    warnings = note_service.validate_note(note)
    assert len(warnings) == 0

def test_validate_note_not_draft_warning(note_service):
    note = ClinicalNote(
        chief_complaint="Cough",
        metadata=NoteMetadata(is_draft=False)
    )
    warnings = note_service.validate_note(note)
    assert any("not marked as draft" in w for w in warnings)
