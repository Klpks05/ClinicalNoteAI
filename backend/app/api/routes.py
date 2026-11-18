import os
import json
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.responses import PlainTextResponse

from backend.app.core.config import settings, REPO_ROOT
from backend.app.models.schemas import (
    AudioUploadResponse,
    TranscribeRequest,
    TranscribeResponse,
    TimestampSegmentSchema,
    UtteranceSchema,
    ReconstructRequest,
    ReconstructResponse,
    GenerateNoteRequest,
    GenerateNoteResponse,
    UpdateNoteRequest,
    SavedNoteRecord,
    ClinicalNoteSchema,
    NoteMetadataSchema,
    EndToEndProcessResponse,
    HealthResponse,
)
from backend.app.utils.audio_utils import (
    validate_audio_file,
    ensure_audio_for_whisper,
    AudioValidationError,
    format_file_size,
    format_duration,
)
from backend.app.services.speech_service import (
    SpeechService,
    SpeechServiceError,
    TimestampedSegment,
)
from backend.app.services.conversation_service import (
    ConversationService,
    ConversationServiceError,
)
from backend.app.services.model_service import (
    ModelService,
    ModelServiceError,
)
from backend.app.services.clinical_note_service import (
    ClinicalNoteService,
    ClinicalNoteServiceError,
    ClinicalNote,
    NoteMetadata,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def serialize_segment_data(s: Any) -> dict:
    """Safely serialize segment data regardless of object type."""
    if isinstance(s, dict):
        return s
    if hasattr(s, "to_dict"):
        return s.to_dict()
    if hasattr(s, "model_dump"):
        return s.model_dump()
    return {
        "text": getattr(s, "text", ""),
        "start": getattr(s, "start", 0.0),
        "end": getattr(s, "end", 0.0),
        "speaker": getattr(s, "speaker", None),
    }

# Services initialization (Singletons or passed dependencies)
speech_service = SpeechService(model_name=settings.WHISPER_MODEL)
conversation_service = ConversationService()
model_service = ModelService()
clinical_note_service = ClinicalNoteService(model_service)

# Helper function to find audio files by file_id
def get_audio_path(file_id: str) -> Path:
    upload_dir = Path(settings.UPLOAD_DIR)
    matches = list(upload_dir.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file with id '{file_id}' not found."
        )
    return matches[0]

# Helper to save notes to disk
def save_note_to_disk(note_id: str, data: dict):
    notes_dir = Path(settings.NOTES_DIR)
    note_path = notes_dir / f"{note_id}.json"
    with open(note_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_note_from_disk(note_id: str) -> dict:
    notes_dir = Path(settings.NOTES_DIR)
    note_path = notes_dir / f"{note_id}.json"
    if not note_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Note with id '{note_id}' not found."
        )
    with open(note_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# SYSTEM HEALTH & STATUS
# ============================================================

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Returns application health and model loading statuses."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        whisper_model_loaded=speech_service.is_loaded,
        whisper_model_name=settings.WHISPER_MODEL,
        clinical_llm_loaded=model_service.is_loaded,
        base_model_name=settings.BASE_MODEL_NAME,
        lora_adapter_path=settings.ADAPTER_PATH,
        device=model_service.device,
    )

@router.post("/models/load")
def load_models():
    """Explicitly pre-loads the Whisper and Clinical LLM models."""
    try:
        if not speech_service.is_loaded:
            speech_service.load_model()
        if not model_service.is_loaded:
            adapter_to_load = settings.ADAPTER_PATH if settings.USE_LORA_ADAPTER else None
            model_service.load_model(
                base_model_name=settings.BASE_MODEL_NAME,
                adapter_path=adapter_to_load,
            )
        return {
            "status": "success",
            "message": "Models loaded successfully into memory",
            "device": model_service.device,
            "has_adapter": model_service.has_adapter,
        }
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model loading failed: {str(e)}"
        )

# ============================================================
# AUDIO UPLOAD & TRANSCRIPTION
# ============================================================

@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload consultation audio for processing."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    
    temp_target = Path(settings.UPLOAD_DIR) / f"{file_id}{ext}"
    try:
        contents = await file.read()
        with open(temp_target, "wb") as f:
            f.write(contents)
            
        info = validate_audio_file(temp_target)
        return AudioUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_size_bytes=info.file_size_bytes,
            duration_seconds=info.duration_seconds,
            format=info.format,
            message="Audio uploaded and validated successfully",
        )
    except AudioValidationError as e:
        if temp_target.exists():
            temp_target.unlink()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {str(e)}")

@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe_audio(payload: TranscribeRequest):
    """Transcribe an uploaded audio file using Whisper."""
    audio_path = get_audio_path(payload.file_id)
    whisper_audio_path, is_temp = ensure_audio_for_whisper(audio_path)
    try:
        if not speech_service.is_loaded:
            speech_service.load_model()
            
        result = speech_service.transcribe(
            audio_path=whisper_audio_path,
            return_timestamps=payload.return_timestamps,
        )
        
        segments = [
            TimestampSegmentSchema(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
            )
            for seg in result.segments
        ]
        
        return TranscribeResponse(
            file_id=payload.file_id,
            text=result.text,
            segments=segments,
            language=result.language,
        )
    except SpeechServiceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        if is_temp and whisper_audio_path.exists():
            whisper_audio_path.unlink(missing_ok=True)

# ============================================================
# CONVERSATION RECONSTRUCTION
# ============================================================

@router.post("/reconstruct", response_model=ReconstructResponse)
def reconstruct_conversation(payload: ReconstructRequest):
    """
    Reconstruct chronological Doctor-Patient dialogue from:
    - A single transcript containing DOCTOR/PATIENT labels or raw text
    - Separate doctor and patient transcript tracks with timestamps
    """
    try:
        if payload.doctor_text and payload.patient_text:
            conv = conversation_service.reconstruct_from_dual_transcripts(
                doctor_text=payload.doctor_text,
                patient_text=payload.patient_text,
                doctor_segments=[serialize_segment_data(s) for s in payload.doctor_segments] if payload.doctor_segments else None,
                patient_segments=[serialize_segment_data(s) for s in payload.patient_segments] if payload.patient_segments else None,
                consultation_id=payload.consultation_id,
            )
        elif payload.text:
            conv = conversation_service.reconstruct_from_single_transcript(
                text=payload.text,
                consultation_id=payload.consultation_id,
            )
        elif payload.file_id:
            # If only file_id provided, transcribe first then reconstruct
            audio_path = get_audio_path(payload.file_id)
            if not speech_service.is_loaded:
                speech_service.load_model()
            result = speech_service.transcribe(audio_path=audio_path)
            conv = conversation_service.reconstruct_from_single_transcript(
                text=result.text,
                consultation_id=payload.consultation_id,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either 'text', 'file_id', or both 'doctor_text' and 'patient_text'."
            )

        utterances = [
            UtteranceSchema(
                speaker=u.speaker,
                text=u.text,
                start_time=u.start_time,
                end_time=u.end_time,
            )
            for u in conv.utterances
        ]

        return ReconstructResponse(
            consultation_id=conv.consultation_id,
            utterances=utterances,
            formatted_conversation=conv.to_formatted_text(),
            total_utterances=len(utterances),
            doctor_utterances_count=len(conv.doctor_utterances),
            patient_utterances_count=len(conv.patient_utterances),
        )
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ============================================================
# CLINICAL NOTE GENERATION
# ============================================================

@router.post("/generate-note", response_model=GenerateNoteResponse)
def generate_note(payload: GenerateNoteRequest):
    """Generate structured clinical note from formatted conversation text."""
    if not payload.conversation or not payload.conversation.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation text cannot be empty."
        )

    try:
        if not model_service.is_loaded:
            adapter_to_load = settings.ADAPTER_PATH if settings.USE_LORA_ADAPTER else None
            model_service.load_model(
                base_model_name=settings.BASE_MODEL_NAME,
                adapter_path=adapter_to_load,
            )

        note = clinical_note_service.generate_note(payload.conversation)
        warnings = clinical_note_service.validate_note(note)

        # Save to disk automatically
        note_id = str(uuid.uuid4())
        record = {
            "note_id": note_id,
            "filename": f"clinical_note_{note_id[:8]}.json",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "consultation_id": payload.consultation_id,
            "clinical_note": note.to_dict(),
            "conversation": payload.conversation,
        }
        save_note_to_disk(note_id, record)

        return GenerateNoteResponse(
            clinical_note=ClinicalNoteSchema(**note.to_dict()),
            warnings=warnings,
        )
    except ClinicalNoteServiceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Note generation failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Generation error: {str(e)}")

# ============================================================
# END-TO-END PROCESSING PIPELINE
# ============================================================

@router.post("/process", response_model=EndToEndProcessResponse)
async def process_full_pipeline(
    file: UploadFile = File(...),
    patient_file: Optional[UploadFile] = File(None),
    consultation_id: Optional[str] = Form(None),
):
    """
    Run complete end-to-end pipeline:
    Audio Upload -> Whisper Transcription -> Conversation Reconstruction -> Clinical Note Generation
    Supports single mixed consultation audio, or separate Doctor and Patient tracks.
    """
    start_time = time.time()
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    doc_target = Path(settings.UPLOAD_DIR) / f"{file_id}_doctor{ext}"
    
    pat_target = None
    if patient_file:
        pat_ext = Path(patient_file.filename).suffix.lower()
        pat_target = Path(settings.UPLOAD_DIR) / f"{file_id}_patient{pat_ext}"

    temp_files_to_cleanup: list[Path] = []

    try:
        # 1. Save and validate files
        doc_contents = await file.read()
        with open(doc_target, "wb") as f:
            f.write(doc_contents)
        temp_files_to_cleanup.append(doc_target)
        validate_audio_file(doc_target)

        if pat_target:
            pat_contents = await patient_file.read()
            with open(pat_target, "wb") as f:
                f.write(pat_contents)
            temp_files_to_cleanup.append(pat_target)
            validate_audio_file(pat_target)

        # 2. Ensure audio is prepared/converted for Whisper and diarization (16kHz mono WAV)
        doc_audio_path, doc_is_temp = ensure_audio_for_whisper(doc_target)
        if doc_is_temp:
            temp_files_to_cleanup.append(doc_audio_path)

        pat_audio_path = None
        if pat_target:
            pat_audio_path, pat_is_temp = ensure_audio_for_whisper(pat_target)
            if pat_is_temp:
                temp_files_to_cleanup.append(pat_audio_path)

        # 3. Ensure models loaded
        if not speech_service.is_loaded:
            speech_service.load_model()
        if not model_service.is_loaded:
            adapter_to_load = settings.ADAPTER_PATH if settings.USE_LORA_ADAPTER else None
            model_service.load_model(
                base_model_name=settings.BASE_MODEL_NAME,
                adapter_path=adapter_to_load,
            )

        # 4. Transcribe
        if pat_audio_path:
            # Dual track transcription
            doc_result = speech_service.transcribe(doc_audio_path, return_timestamps=True)
            pat_result = speech_service.transcribe(pat_audio_path, return_timestamps=True)
            
            for s in doc_result.segments:
                s.speaker = "Doctor"
            for s in pat_result.segments:
                s.speaker = "Patient"

            combined_text = f"Doctor: {doc_result.text}\nPatient: {pat_result.text}"
            all_segments = [
                TimestampSegmentSchema(text=f"[Doctor] {s.text}", start=s.start, end=s.end, speaker="Doctor")
                for s in doc_result.segments
            ] + [
                TimestampSegmentSchema(text=f"[Patient] {s.text}", start=s.start, end=s.end, speaker="Patient")
                for s in pat_result.segments
            ]
            all_segments.sort(key=lambda s: s.start)
            
            transcription_resp = TranscribeResponse(
                file_id=file_id,
                text=combined_text,
                segments=all_segments,
            )

            # Reconstruct conversation chronologically
            conv = conversation_service.reconstruct_from_dual_transcripts(
                doctor_text=doc_result.text,
                patient_text=pat_result.text,
                doctor_segments=[s.to_dict() for s in doc_result.segments],
                patient_segments=[s.to_dict() for s in pat_result.segments],
                consultation_id=consultation_id,
            )
        else:
            # Single track transcription with acoustic speaker diarization
            single_result = speech_service.transcribe(doc_audio_path, return_timestamps=True)
            single_segments = [s.to_dict() for s in single_result.segments]
            
            conv = conversation_service.reconstruct_from_single_transcript(
                text=single_result.text,
                consultation_id=consultation_id,
                segments=single_segments,
                audio_path=doc_audio_path,
            )

            # Map diarized speaker labels to segments if 1-to-1 matching
            if len(conv.utterances) == len(single_result.segments):
                for s, u in zip(single_result.segments, conv.utterances):
                    s.speaker = u.speaker

            segments = [
                TimestampSegmentSchema(
                    text=s.text,
                    start=s.start,
                    end=s.end,
                    speaker=getattr(s, "speaker", None),
                )
                for s in single_result.segments
            ]
            transcription_resp = TranscribeResponse(
                file_id=file_id,
                text=single_result.text,
                segments=segments,
            )

        conv_utterances = [
            UtteranceSchema(
                speaker=u.speaker,
                text=u.text,
                start_time=u.start_time,
                end_time=u.end_time,
            )
            for u in conv.utterances
        ]
        conversation_resp = ReconstructResponse(
            consultation_id=conv.consultation_id,
            utterances=conv_utterances,
            formatted_conversation=conv.to_formatted_text(),
            total_utterances=len(conv_utterances),
            doctor_utterances_count=len(conv.doctor_utterances),
            patient_utterances_count=len(conv.patient_utterances),
        )

        # 5. Generate Clinical Note
        formatted_dialogue = conv.to_formatted_text()
        note = clinical_note_service.generate_note(formatted_dialogue)
        warnings = clinical_note_service.validate_note(note)

        note_id = str(uuid.uuid4())
        record = {
            "note_id": note_id,
            "filename": f"note_{file.filename}.json",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "consultation_id": consultation_id,
            "clinical_note": note.to_dict(),
            "conversation": formatted_dialogue,
        }
        save_note_to_disk(note_id, record)

        elapsed = time.time() - start_time
        return EndToEndProcessResponse(
            file_id=file_id,
            filename=file.filename,
            transcription=transcription_resp,
            conversation=conversation_resp,
            clinical_note=ClinicalNoteSchema(**note.to_dict()),
            note_id=note_id,
            warnings=warnings,
            processing_time_seconds=round(elapsed, 2),
        )
    except AudioValidationError as e:
        logger.warning(f"Audio validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except (SpeechServiceError, ModelServiceError, ConversationServiceError, ClinicalNoteServiceError) as e:
        logger.error(f"Processing service failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"End-to-end processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"End-to-end processing error: {str(e)}"
        )
    finally:
        for tf in temp_files_to_cleanup:
            try:
                if tf.exists():
                    tf.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {tf}: {e}")


# ============================================================
# NOTE PERSISTENCE & MANAGEMENT
# ============================================================

@router.get("/notes", response_model=list[SavedNoteRecord])
def list_notes():
    """List all saved clinical notes."""
    notes_dir = Path(settings.NOTES_DIR)
    records = []
    for file_path in sorted(notes_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append(SavedNoteRecord(**data))
        except Exception as e:
            logger.warning(f"Failed to read note {file_path.name}: {e}")
    return records

@router.get("/notes/{note_id}", response_model=SavedNoteRecord)
def get_note(note_id: str):
    """Retrieve a specific clinical note by ID."""
    data = load_note_from_disk(note_id)
    return SavedNoteRecord(**data)

@router.put("/notes/{note_id}", response_model=SavedNoteRecord)
def update_note(note_id: str, payload: UpdateNoteRequest):
    """Update/edit sections of a generated clinical note."""
    data = load_note_from_disk(note_id)
    note_dict = data.get("clinical_note", {})

    # Apply updates
    updates = payload.model_dump(exclude_unset=True)
    for field, val in updates.items():
        if val is not None:
            note_dict[field] = val

    data["clinical_note"] = note_dict
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_note_to_disk(note_id, data)

    return SavedNoteRecord(**data)

@router.delete("/notes/{note_id}")
def delete_note(note_id: str):
    """Delete a saved clinical note."""
    notes_dir = Path(settings.NOTES_DIR)
    note_path = notes_dir / f"{note_id}.json"
    if not note_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    note_path.unlink()
    return {"status": "success", "message": f"Note {note_id} deleted."}

@router.get("/notes/{note_id}/export")
def export_note_text(note_id: str, format: str = "text"):
    """Export a clinical note in plain text format or JSON."""
    data = load_note_from_disk(note_id)
    if format == "json":
        return data

    clinical_note_dict = data.get("clinical_note", {})
    metadata_dict = clinical_note_dict.get("metadata", {})
    
    note_obj = ClinicalNote(
        chief_complaint=clinical_note_dict.get("chief_complaint", "Not mentioned."),
        history_of_present_illness=clinical_note_dict.get("history_of_present_illness", "Not mentioned."),
        past_medical_history=clinical_note_dict.get("past_medical_history", "Not mentioned."),
        medications=clinical_note_dict.get("medications", "Not mentioned."),
        allergies=clinical_note_dict.get("allergies", "Not mentioned."),
        review_of_systems=clinical_note_dict.get("review_of_systems", "Not mentioned."),
        examination=clinical_note_dict.get("examination", "Not mentioned."),
        assessment=clinical_note_dict.get("assessment", "Not mentioned."),
        plan=clinical_note_dict.get("plan", "Not mentioned."),
        metadata=NoteMetadata(
            generated_at=metadata_dict.get("generated_at", ""),
            model_name=metadata_dict.get("model_name", ""),
            adapter_name=metadata_dict.get("adapter_name", ""),
        ),
    )

    return PlainTextResponse(
        content=note_obj.to_text(),
        headers={"Content-Disposition": f"attachment; filename=clinical_note_{note_id[:8]}.txt"}
    )

# ============================================================
# PRIMOCK57 DEMO ENDPOINT
# ============================================================

@router.post("/demo/primock57", response_model=EndToEndProcessResponse)
def run_primock57_demo(day: int = 1, consultation: int = 1):
    """
    Run an end-to-end demo using an existing PriMock57 consultation
    (reconstructing from actual TextGrids and generating note with LoRA).
    """
    start_time = time.time()
    consultation_id = f"day{day}_consultation{consultation:02d}"

    doc_tg = REPO_ROOT / "data" / "primock57" / "transcripts" / f"{consultation_id}_doctor.TextGrid"
    pat_tg = REPO_ROOT / "data" / "primock57" / "transcripts" / f"{consultation_id}_patient.TextGrid"

    if not doc_tg.exists() or not pat_tg.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PriMock57 files for {consultation_id} not found."
        )

    # 1. Reconstruct conversation from TextGrids
    conv = conversation_service.reconstruct_from_textgrids(
        doctor_textgrid=doc_tg,
        patient_textgrid=pat_tg,
        consultation_id=consultation_id,
    )
    formatted_dialogue = conv.to_formatted_text()

    # 2. Ensure model loaded
    if not model_service.is_loaded:
        adapter_to_load = settings.ADAPTER_PATH if settings.USE_LORA_ADAPTER else None
        model_service.load_model(
            base_model_name=settings.BASE_MODEL_NAME,
            adapter_path=adapter_to_load,
        )

    # 3. Generate Clinical Note
    note = clinical_note_service.generate_note(formatted_dialogue)
    warnings = clinical_note_service.validate_note(note)

    # 4. Save note
    note_id = str(uuid.uuid4())
    record = {
        "note_id": note_id,
        "filename": f"demo_{consultation_id}.json",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "consultation_id": consultation_id,
        "clinical_note": note.to_dict(),
        "conversation": formatted_dialogue,
    }
    save_note_to_disk(note_id, record)

    conv_utterances = [
        UtteranceSchema(
            speaker=u.speaker,
            text=u.text,
            start_time=u.start_time,
            end_time=u.end_time,
        )
        for u in conv.utterances
    ]

    elapsed = time.time() - start_time
    return EndToEndProcessResponse(
        file_id=consultation_id,
        filename=f"{consultation_id}.wav",
        transcription=TranscribeResponse(
            file_id=consultation_id,
            text=formatted_dialogue,
            segments=[
                TimestampSegmentSchema(
                    text=f"[{u.speaker}] {u.text}",
                    start=u.start_time,
                    end=u.end_time,
                    speaker=u.speaker,
                )
                for u in conv.utterances
            ],
        ),
        conversation=ReconstructResponse(
            consultation_id=consultation_id,
            utterances=conv_utterances,
            formatted_conversation=formatted_dialogue,
            total_utterances=len(conv_utterances),
            doctor_utterances_count=len(conv.doctor_utterances),
            patient_utterances_count=len(conv.patient_utterances),
        ),
        clinical_note=ClinicalNoteSchema(**note.to_dict()),
        note_id=note_id,
        warnings=warnings,
        processing_time_seconds=round(elapsed, 2),
    )
