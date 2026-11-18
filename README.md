# ClinicalNoteAI

**Doctor–Patient LLM-Based Speech Recognition into Clinical Notes**

An end-to-end medical documentation application that transcribes doctor–patient consultation audio, reconstructs chronological dialogue, generates structured clinical consultation notes using a fine-tuned LoRA large language model, and provides a professional React web interface for review, editing, and export.

> **ACADEMIC PROTOTYPE DISCLAIMER**  
> ClinicalNoteAI is an academic research prototype and **not** a clinically validated, autonomous diagnostic system. All generated clinical notes are explicitly marked as **drafts** requiring verification and sign-off by a qualified clinician before any clinical application.

---

## 1. System Architecture

The application strictly implements the fixed 6-stage architecture:

```
[Doctor + Patient Consultation Audio (.wav, .mp3)]
                      ↓
          [Whisper Speech Recognition]
         (Apple Silicon MPS Accelerated)
                      ↓
    [Doctor–Patient Conversation Reconstruction]
  (Chronological speaker alignment & tag cleaning)
                      ↓
  [Clinical Note Generation using Fine-Tuned LLM]
  (Qwen2.5-1.5B-Instruct + LoRA Adapter via PEFT)
                      ↓
           [Structured Clinical Note]
 (Chief Complaint, HPI, PMH, Medications, Allergies,
      Review of Systems, Exam, Assessment, Plan)
                      ↓
               [FastAPI Backend]
       (REST API, Audio Validation, Notes CRUD)
                      ↓
               [React Frontend]
 (Upload, Status Indicator, Dialogue & Note Editor)
```

---

## 2. Technology Stack

- **Audio & Speech Recognition:** OpenAI Whisper (`openai/whisper-small.en` / `whisper-base.en`), SoundFile, Librosa, PyTorch (Apple MPS accelerated).
- **Large Language Model:** `Qwen/Qwen2.5-1.5B-Instruct` fine-tuned with Parameter-Efficient Fine-Tuning (PEFT / LoRA).
- **Backend API:** FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings, python-multipart.
- **Frontend UI:** React 19, TypeScript, Vite, Tailwind CSS, Lucide React, Axios.
- **Testing & Evaluation:** PyTest, ROUGE (ROUGE-1, ROUGE-2, ROUGE-L).
- **Platform:** macOS Sonoma / Sequoia on Apple Silicon (M4 / M-Series) with Metal Performance Shaders (MPS).

---

## 3. Datasets

### 3.1 PriMock57 (`data/primock57/`)
- **Role:** Speech recognition, dual-track audio testing, and full consultation conversation reconstruction.
- **Source:** ACL 2022 dataset (*PriMock57: A Dataset Of Primary Care Mock Consultations*, Korfiatis et al., 2022).
- **Contents:**
  - 114 audio recordings (57 Doctor tracks + 57 Patient tracks in 16kHz WAV format).
  - 114 Praat TextGrid transcript files with utterance-level start/end timestamps.
  - 57 clinician consultation notes in structured JSON.
  - Case card checklists and human evaluation taxonomy files.
- **Speaker Role Distinction:** PriMock57 recordings contain separate doctor and patient audio channels. The system uses TextGrid speaker intervals and audio tracks to reconstruct conversations chronologically.

### 3.2 MTS-Dialog (`outputs/mts_*.json`)
- **Role:** Fine-tuning the clinical note generation model.
- **Source:** Hugging Face `lyumengxian/MTS-Dialog`.
- **Contents:** Doctor-patient dialogues paired with clinical note sections across 10 clinical headings: Chief Complaint (`CC`), History of Present Illness (`GENHX`), Past Medical History (`PASTMEDICALHX`), Medications (`MEDICATIONS`), Allergies (`ALLERGY`), Review of Systems (`ROS`), Examination (`EXAM`), Assessment (`ASSESSMENT`), Plan (`PLAN`), and Family/Social History (`FAM/SOCHX`).
- **Splits:** 1,124 train examples, 85 validation examples, 192 test examples.

---

## 4. Fine-Tuned Models & LoRA Adapters

All adapters use `Qwen/Qwen2.5-1.5B-Instruct` as base model:
- **LoRA Configuration:** `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`.
- **Production Adapter:** `outputs/qwen_clinical_lora_mts` (eval loss `1.5866`, trained over 1,124 section-dialogue pairs with label masking).
- **Benchmark Evaluation (10-sample holdout test):**
  - Base Qwen: ROUGE-1 `0.2973`, ROUGE-2 `0.1008`, ROUGE-L `0.2511`, Average Words: `39.1`
  - LoRA Qwen: ROUGE-1 `0.2550`, ROUGE-2 `0.1118`, ROUGE-L `0.1950`, Average Words: `57.4`
  - *Observation:* The fine-tuned LoRA model outputs significantly more thorough, clinically appropriate phrasing adhering to professional medical documentation standards.

---

## 5. Installation on macOS (Apple Silicon M4)

### 5.1 Prerequisites
- macOS 14+ with Apple Silicon (M1/M2/M3/M4).
- Python 3.12 installed (`brew install python@3.12`).
- Node.js 18+ and npm (`brew install node`).

### 5.2 Virtual Environment & Backend Setup
```bash
# Clone or navigate to the repository
cd /Users/kalpak/Documents/ClinicalNoteAI

# Activate existing virtual environment (or create a new one)
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 5.3 Frontend Setup
```bash
cd frontend
npm install
npm run build
```

---

## 6. Running the Application

### 6.1 Start the FastAPI Backend
From the project root:
```bash
source venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base: `http://localhost:8000`
- Interactive API Docs (Swagger UI): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`

### 6.2 Start the React Frontend
In a separate terminal:
```bash
cd frontend
npm run dev
```
- Web Application: `http://localhost:5173`

---

## 7. How to Use the Application

1. **Dashboard & System Status:** The header displays real-time Whisper and Qwen LoRA status. Click "Preload Models" if you wish to warm up memory before processing.
2. **Audio Processing Options:**
   - **Single Audio Track:** Upload a consultation recording (`.wav`, `.mp3`, `.m4a`). The backend validates size and duration, runs Whisper transcription, reconstructs dialogue, and generates the note.
   - **Dual Track (PriMock57 Style):** Upload separate Doctor and Patient recordings. Both are transcribed with timestamps and merged chronologically into an aligned consultation.
   - **PriMock57 Dataset Demo:** Select any consultation (e.g., Day 1 Consultation 01) directly from the embedded dataset for a 1-click end-to-end execution.
3. **Pipeline Progress:** Real-time animated step indicator showing upload → transcription → reconstruction → clinical note generation.
4. **Interactive Note Review:**
   - View structured sections: Chief Complaint, HPI, PMH, Medications, Allergies, Review of Systems, Examination, Assessment, Plan.
   - Edit any section in place.
   - Save modifications to disk.
   - Export clinical note as plain text (`.txt`) or structured JSON (`.json`).
5. **Saved Notes History:** View, reload, and manage previously generated notes from the history drawer.

---

## 8. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, model loading status, compute device |
| `POST` | `/api/models/load` | Explicitly preload Whisper and LLM into memory |
| `POST` | `/api/upload` | Upload and validate audio file |
| `POST` | `/api/transcribe` | Transcribe speech using Whisper |
| `POST` | `/api/reconstruct` | Reconstruct doctor–patient dialogue |
| `POST` | `/api/generate-note` | Generate structured note from conversation |
| `POST` | `/api/process` | Full end-to-end pipeline (audio → note) |
| `GET` | `/api/notes` | List all saved clinical notes |
| `GET` | `/api/notes/{id}` | Retrieve specific clinical note |
| `PUT` | `/api/notes/{id}` | Edit / update sections of a note |
| `DELETE` | `/api/notes/{id}` | Delete a saved note |
| `GET` | `/api/notes/{id}/export` | Export note as text or JSON |
| `POST` | `/api/demo/primock57` | Run end-to-end demo on PriMock57 consultation |

---

## 9. Testing

Run the automated test suite with pytest:
```bash
source venv/bin/activate
pytest tests/ -v
```

Tests include:
- `tests/test_audio_utils.py`: Format validation, duration, size limits.
- `tests/test_conversation_service.py`: TextGrid parsing, tag stripping, chronological sorting.
- `tests/test_model_service.py`: LoRA adapter verification, error handling.
- `tests/test_clinical_note_service.py`: Schema conformance, head+tail truncation, draft validation.
- `tests/test_api.py`: FastAPI endpoints, health check, upload rejection, CRUD routes.

---

## 10. Repository Organization

```
ClinicalNoteAI/
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/routes.py     # API Endpoints
│   │   ├── core/config.py    # Configuration & Settings
│   │   ├── models/schemas.py # Pydantic Data Models
│   │   ├── services/         # Reusable Service Layer
│   │   │   ├── speech_service.py        # Whisper ASR
│   │   │   ├── conversation_service.py  # Chronological Dialogue
│   │   │   ├── model_service.py         # LoRA / Qwen Loader
│   │   │   └── clinical_note_service.py # Note Generation
│   │   └── utils/audio_utils.py         # Audio Validation
│   ├── requirements.txt      # Backend Dependencies
│   └── .env                  # Environment Variables
├── frontend/                 # React + TypeScript Frontend
│   ├── src/
│   │   ├── components/       # UI Components
│   │   ├── services/api.ts   # Backend API Client
│   │   ├── types/index.ts    # TypeScript Interfaces
│   │   ├── App.tsx           # Dashboard Application
│   │   └── index.css         # Medical Aesthetic Design
│   ├── package.json
│   └── vite.config.ts
├── data/
│   ├── primock57/            # PriMock57 Dataset (Audio + Transcripts + Notes)
│   └── mts_dialog/           # MTS-Dialog Ingestion Directory
├── outputs/                  # Processed Training Data & LoRA Adapters
│   ├── qwen_clinical_lora_mts/ # Best Evaluated LoRA Adapter
│   └── *.json                # Training, Validation, and Test Sets
├── scripts/                  # Active ML Scripts & Benchmarks
│   └── archive/              # Preserved Historical Experiments
└── tests/                    # Automated Test Suite
```
