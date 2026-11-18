"""
Comprehensive verification script for ClinicalNoteAI.
Tests all three modes:
1. PriMock57 dataset demo.
2. External single-track audio.
3. External dual-track doctor and patient audio (longer recording).
"""

import sys
import time
import wave
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)

def test_mode_1_primock_demo():
    print("\n" + "=" * 70)
    print("MODE 1: PriMock57 Dataset Demo")
    print("=" * 70)
    t0 = time.time()
    resp = client.post("/api/demo/primock57?day=1&consultation=1")
    elapsed = time.time() - t0
    assert resp.status_code == 200, f"Demo failed ({resp.status_code}): {resp.text}"
    data = resp.json()

    print(f"Status: SUCCESS ({elapsed:.2f}s)")
    print(f"File ID: {data['file_id']}")
    print(f"Total Segments: {len(data['transcription']['segments'])}")
    print(f"Total Utterances: {data['conversation']['total_utterances']}")
    print(f"Doctor Utterances: {data['conversation']['doctor_utterances_count']}")
    print(f"Patient Utterances: {data['conversation']['patient_utterances_count']}")
    print(f"Chief Complaint: {data['clinical_note']['chief_complaint']}")
    print(f"HPI: {data['clinical_note']['history_of_present_illness'][:100]}...")

    metadata = data['clinical_note']['metadata']
    if settings.USE_LORA_ADAPTER:
        assert "qwen_clinical_lora_mts" in metadata.get('adapter_name', '')
    else:
        assert metadata.get('adapter_name') in ("none", "", None)

    # Validate segment schema
    first_seg = data['transcription']['segments'][0]
    assert "start" in first_seg and "end" in first_seg and "text" in first_seg
    assert first_seg.get("speaker") in ("Doctor", "Patient")
    print("Segment schema verified: start, end, text, speaker all present.")
    return data

def test_mode_2_single_track():
    print("\n" + "=" * 70)
    print("MODE 2: External Single-Track Audio Upload")
    print("=" * 70)

    # Extract 30 seconds from day1_consultation01_doctor.wav as external single-track recording
    src_wav = REPO_ROOT / "data" / "primock57" / "audio" / "day1_consultation01_doctor.wav"
    assert src_wav.exists(), f"Source audio missing: {src_wav}"

    temp_single = REPO_ROOT / "uploads" / "external_single_test.wav"
    with wave.open(str(src_wav), "rb") as wf_in:
        sr = wf_in.getframerate()
        n_frames = min(wf_in.getnframes(), sr * 30)  # 30 seconds
        frames = wf_in.readframes(n_frames)
        with wave.open(str(temp_single), "wb") as wf_out:
            wf_out.setnchannels(wf_in.getnchannels())
            wf_out.setsampwidth(wf_in.getsampwidth())
            wf_out.setframerate(sr)
            wf_out.writeframes(frames)

    print(f"Created 30s single-track external audio: {temp_single}")

    t0 = time.time()
    with open(temp_single, "rb") as f:
        resp = client.post(
            "/api/process",
            files={"file": ("external_single_consultation.wav", f, "audio/wav")},
        )
    elapsed = time.time() - t0

    # Clean up local test file
    temp_single.unlink(missing_ok=True)

    assert resp.status_code == 200, f"Single track process failed ({resp.status_code}): {resp.text}"
    data = resp.json()

    print(f"Status: SUCCESS ({elapsed:.2f}s)")
    print(f"File ID: {data['file_id']}")
    print(f"Transcription Text Length: {len(data['transcription']['text'])} chars")
    print(f"Total Segments: {len(data['transcription']['segments'])}")
    print(f"Total Utterances: {data['conversation']['total_utterances']}")
    print(f"Doctor Utterances: {data['conversation']['doctor_utterances_count']}")
    print(f"Patient Utterances: {data['conversation']['patient_utterances_count']}")
    print(f"Chief Complaint: {data['clinical_note']['chief_complaint']}")
    print(f"HPI: {data['clinical_note']['history_of_present_illness'][:100]}...")

    metadata = data['clinical_note']['metadata']
    if settings.USE_LORA_ADAPTER:
        assert "qwen_clinical_lora_mts" in metadata.get('adapter_name', '')
    else:
        assert metadata.get('adapter_name') in ("none", "", None)

    # Verify segments
    assert len(data['transcription']['segments']) > 0
    seg0 = data['transcription']['segments'][0]
    print(f"Sample segment: start={seg0['start']}, end={seg0['end']}, speaker={seg0.get('speaker')}, text='{seg0['text']}'")
    return data

def test_mode_3_dual_track_long():
    print("\n" + "=" * 70)
    print("MODE 3: External Dual-Track (Doctor + Patient) Longer Audio Recording")
    print("=" * 70)

    doc_wav = REPO_ROOT / "data" / "primock57" / "audio" / "day1_consultation01_doctor.wav"
    pat_wav = REPO_ROOT / "data" / "primock57" / "audio" / "day1_consultation01_patient.wav"

    # Extract 60 seconds of dual-track audio (longer external WAV recording)
    doc_temp = REPO_ROOT / "uploads" / "external_dual_doctor_60s.wav"
    pat_temp = REPO_ROOT / "uploads" / "external_dual_patient_60s.wav"

    for src, dst in [(doc_wav, doc_temp), (pat_wav, pat_temp)]:
        with wave.open(str(src), "rb") as wf_in:
            sr = wf_in.getframerate()
            n_frames = min(wf_in.getnframes(), sr * 60)  # 60s
            frames = wf_in.readframes(n_frames)
            with wave.open(str(dst), "wb") as wf_out:
                wf_out.setnchannels(wf_in.getnchannels())
                wf_out.setsampwidth(wf_in.getsampwidth())
                wf_out.setframerate(sr)
                wf_out.writeframes(frames)

    print(f"Created 60s dual-track external audio files: {doc_temp.name}, {pat_temp.name}")

    t0 = time.time()
    with open(doc_temp, "rb") as f_doc, open(pat_temp, "rb") as f_pat:
        resp = client.post(
            "/api/process",
            files={
                "file": ("external_doctor_track.wav", f_doc, "audio/wav"),
                "patient_file": ("external_patient_track.wav", f_pat, "audio/wav"),
            },
        )
    elapsed = time.time() - t0

    # Clean up local test files
    doc_temp.unlink(missing_ok=True)
    pat_temp.unlink(missing_ok=True)

    assert resp.status_code == 200, f"Dual track process failed ({resp.status_code}): {resp.text}"
    data = resp.json()

    print(f"Status: SUCCESS ({elapsed:.2f}s)")
    print(f"File ID: {data['file_id']}")
    print(f"Transcription Text Length: {len(data['transcription']['text'])} chars")
    print(f"Total Segments: {len(data['transcription']['segments'])}")
    print(f"Total Utterances: {data['conversation']['total_utterances']}")
    print(f"Doctor Utterances: {data['conversation']['doctor_utterances_count']}")
    print(f"Patient Utterances: {data['conversation']['patient_utterances_count']}")
    print(f"Chief Complaint: {data['clinical_note']['chief_complaint']}")
    print(f"Assessment: {data['clinical_note']['assessment']}")
    print(f"Plan: {data['clinical_note']['plan']}")

    metadata = data['clinical_note']['metadata']
    if settings.USE_LORA_ADAPTER:
        assert "qwen_clinical_lora_mts" in metadata.get('adapter_name', '')
    else:
        assert metadata.get('adapter_name') in ("none", "", None)

    # Verify chronological ordering of dual track
    utts = data['conversation']['utterances']
    assert len(utts) >= 2
    for i in range(len(utts) - 1):
        assert utts[i]['start_time'] <= utts[i + 1]['start_time'], "Utterances must be sorted chronologically!"
    print("Chronological ordering verified across dual tracks.")

    return data

if __name__ == "__main__":
    print("Starting Comprehensive Verification Across All 3 Modes...")
    m1 = test_mode_1_primock_demo()
    m2 = test_mode_2_single_track()
    m3 = test_mode_3_dual_track_long()
    print("\n" + "=" * 70)
    print("ALL 3 MODES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
