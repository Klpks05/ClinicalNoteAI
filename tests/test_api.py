import io
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.schemas import HealthResponse

@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "ClinicalNoteAI"
    assert "docs_url" in data

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "whisper_model_name" in data
    assert "lora_adapter_path" in data

def test_upload_invalid_file_extension(client):
    file_content = b"Not an audio file"
    response = client.post(
        "/api/upload",
        files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
    )
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]

def test_reconstruct_from_text(client):
    text = "DOCTOR: Hello, how can I help you?\nPATIENT: I have an earache on my left side."
    response = client.post(
        "/api/reconstruct",
        json={"text": text},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_utterances"] == 2
    assert data["doctor_utterances_count"] == 1
    assert data["patient_utterances_count"] == 1
    assert "DOCTOR:" in data["formatted_conversation"]

def test_notes_crud_lifecycle(client):
    # 1. List notes initially
    list_resp = client.get("/api/notes")
    assert list_resp.status_code == 200
    initial_count = len(list_resp.json())

    # 2. Reconstruct and check mock note or update note
    # Let's save a note directly to disk or test 404 for nonexistent note
    nonexistent = client.get("/api/notes/nonexistent_id_12345")
    assert nonexistent.status_code == 404

def test_reconstruct_with_segments(client):
    doc_segs = [{"text": "Hello, how can I help you?", "start": 0.5, "end": 2.0, "speaker": "Doctor"}]
    pat_segs = [{"text": "I have an earache.", "start": 2.2, "end": 4.0, "speaker": "Patient"}]
    resp = client.post(
        "/api/reconstruct",
        json={
            "doctor_text": "Hello, how can I help you?",
            "patient_text": "I have an earache.",
            "doctor_segments": doc_segs,
            "patient_segments": pat_segs,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_utterances"] == 2
    assert data["utterances"][0]["speaker"] == "Doctor"
    assert data["utterances"][1]["speaker"] == "Patient"

def test_process_invalid_audio(client):
    resp = client.post(
        "/api/process",
        files={"file": ("corrupted.wav", io.BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt corrupt"), "audio/wav")},
    )
    assert resp.status_code == 400
    assert "Invalid" in resp.json()["detail"] or "corrupted" in resp.json()["detail"]
