import axios from 'axios';
import type {
  HealthResponse,
  AudioUploadResponse,
  TranscribeResponse,
  ReconstructResponse,
  ClinicalNote,
  SavedNoteRecord,
  EndToEndProcessResponse,
} from '../types';

const API_BASE_URL = '/api';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health & Models
  async getHealth(): Promise<HealthResponse> {
    const res = await client.get<HealthResponse>('/health');
    return res.data;
  },

  async loadModels(): Promise<{ status: string; message: string; device: string; has_adapter: boolean }> {
    const res = await client.post('/models/load');
    return res.data;
  },

  // Audio Upload
  async uploadAudio(file: File): Promise<AudioUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await client.post<AudioUploadResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  // Transcription
  async transcribeAudio(fileId: string, returnTimestamps: boolean = true): Promise<TranscribeResponse> {
    const res = await client.post<TranscribeResponse>('/transcribe', {
      file_id: fileId,
      return_timestamps: returnTimestamps,
    });
    return res.data;
  },

  // Conversation Reconstruction
  async reconstructConversation(payload: {
    file_id?: string;
    text?: string;
    doctor_text?: string;
    patient_text?: string;
    consultation_id?: string;
  }): Promise<ReconstructResponse> {
    const res = await client.post<ReconstructResponse>('/reconstruct', payload);
    return res.data;
  },

  // Clinical Note Generation
  async generateClinicalNote(conversation: string, consultationId?: string): Promise<{ clinical_note: ClinicalNote; warnings: string[] }> {
    const res = await client.post('/generate-note', {
      conversation,
      consultation_id: consultationId,
    });
    return res.data;
  },

  // Full Pipeline
  async processEndToEnd(
    file: File,
    patientFile?: File,
    consultationId?: string
  ): Promise<EndToEndProcessResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (patientFile) {
      formData.append('patient_file', patientFile);
    }
    if (consultationId) {
      formData.append('consultation_id', consultationId);
    }

    const res = await client.post<EndToEndProcessResponse>('/process', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 min timeout for model inference
    });
    return res.data;
  },

  // Saved Notes CRUD
  async listNotes(): Promise<SavedNoteRecord[]> {
    const res = await client.get<SavedNoteRecord[]>('/notes');
    return res.data;
  },

  async getNote(noteId: string): Promise<SavedNoteRecord> {
    const res = await client.get<SavedNoteRecord>(`/notes/${noteId}`);
    return res.data;
  },

  async updateNote(noteId: string, sections: Partial<ClinicalNote>): Promise<SavedNoteRecord> {
    const res = await client.put<SavedNoteRecord>(`/notes/${noteId}`, sections);
    return res.data;
  },

  async deleteNote(noteId: string): Promise<{ status: string; message: string }> {
    const res = await client.delete(`/notes/${noteId}`);
    return res.data;
  },

  // PriMock57 Demo
  async runPriMockDemo(day: number = 1, consultation: number = 1): Promise<EndToEndProcessResponse> {
    const res = await client.post<EndToEndProcessResponse>(
      `/demo/primock57?day=${day}&consultation=${consultation}`,
      {},
      { timeout: 300000 }
    );
    return res.data;
  },
};
