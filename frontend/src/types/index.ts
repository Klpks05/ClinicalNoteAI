export interface TimestampSegment {
  text: string;
  start: number;
  end: number;
}

export interface AudioUploadResponse {
  file_id: string;
  filename: string;
  file_size_bytes: number;
  duration_seconds: number | null;
  format: string;
  message: string;
}

export interface TranscribeResponse {
  file_id: string;
  text: string;
  segments: TimestampSegment[];
  language: string;
}

export interface Utterance {
  speaker: string; // "Doctor" | "Patient"
  text: string;
  start_time: number;
  end_time: number;
}

export interface ReconstructResponse {
  consultation_id?: string | null;
  utterances: Utterance[];
  formatted_conversation: string;
  total_utterances: number;
  doctor_utterances_count: number;
  patient_utterances_count: number;
}

export interface NoteMetadata {
  generated_at: string;
  model_name: string;
  adapter_name: string;
  is_draft: boolean;
  disclaimer: string;
}

export interface ClinicalNote {
  chief_complaint: string;
  history_of_present_illness: string;
  past_medical_history: string;
  medications: string;
  allergies: string;
  review_of_systems?: string;
  examination: string;
  assessment: string;
  plan: string;
  raw_markdown?: string | null;
  metadata: NoteMetadata;
}

export interface SavedNoteRecord {
  note_id: string;
  filename: string;
  created_at: string;
  updated_at: string;
  consultation_id?: string | null;
  clinical_note: ClinicalNote;
  conversation?: string | null;
}

export interface EndToEndProcessResponse {
  file_id: string;
  filename: string;
  transcription: TranscribeResponse;
  conversation: ReconstructResponse;
  clinical_note: ClinicalNote;
  note_id: string;
  warnings: string[];
  processing_time_seconds: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  whisper_model_loaded: boolean;
  whisper_model_name: string;
  clinical_llm_loaded: boolean;
  base_model_name: string;
  lora_adapter_path: string;
  device: string;
}
