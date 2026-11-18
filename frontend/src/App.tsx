import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { AudioUpload } from './components/AudioUpload';
import { ProcessingStatus, type ProcessingStep } from './components/ProcessingStatus';
import { TranscriptionView } from './components/TranscriptionView';
import { ConversationView } from './components/ConversationView';
import { ClinicalNoteEditor } from './components/ClinicalNoteEditor';
import { SavedNotesList } from './components/SavedNotesList';
import { api } from './services/api';
import type {
  HealthResponse,
  TranscribeResponse,
  ReconstructResponse,
  ClinicalNote,
  SavedNoteRecord,
} from './types';
import { History, ChevronDown, ChevronRight, MessageSquare, FileAudio } from 'lucide-react';

export const App: React.FC = () => {
  // System health
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(false);

  // Pipeline execution state
  const [processingStep, setProcessingStep] = useState<ProcessingStep>('idle');
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | undefined>(undefined);

  // Active outputs
  const [activeTranscription, setActiveTranscription] = useState<TranscribeResponse | null>(null);
  const [activeConversation, setActiveConversation] = useState<ReconstructResponse | null>(null);
  const [activeNote, setActiveNote] = useState<ClinicalNote | null>(null);
  const [activeNoteId, setActiveNoteId] = useState<string | undefined>(undefined);

  // Collapsible supporting views
  const [showConversation, setShowConversation] = useState<boolean>(false);
  const [showTranscription, setShowTranscription] = useState<boolean>(false);

  // Saved notes
  const [savedNotes, setSavedNotes] = useState<SavedNoteRecord[]>([]);
  const [showHistory, setShowHistory] = useState<boolean>(false);

  const fetchHealth = async () => {
    try {
      const data = await api.getHealth();
      setHealth(data);
    } catch (e) {
      console.warn('Backend not yet reachable:', e);
    }
  };

  const fetchSavedNotes = async () => {
    try {
      const data = await api.listNotes();
      setSavedNotes(data);
    } catch (e) {
      console.warn('Failed to load saved notes:', e);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchSavedNotes();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handlePreloadModels = async () => {
    setIsLoadingModels(true);
    try {
      await api.loadModels();
      await fetchHealth();
    } catch (e: any) {
      alert(`Model loading error: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setIsLoadingModels(false);
    }
  };

  // Run full pipeline on a single file
  const handleProcessSingle = async (file: File) => {
    setProcessingStep('uploading');
    setProcessingError(null);
    setProcessingTime(undefined);

    try {
      setProcessingStep('transcribing');
      const res = await api.processEndToEnd(file);
      setProcessingStep('reconstructing');
      setActiveTranscription(res.transcription);
      setActiveConversation(res.conversation);

      setProcessingStep('generating');
      setActiveNote(res.clinical_note);
      setActiveNoteId(res.note_id);
      setProcessingTime(res.processing_time_seconds);

      setProcessingStep('complete');
      await fetchSavedNotes();
    } catch (e: any) {
      setProcessingStep('error');
      setProcessingError(e?.response?.data?.detail || e.message || 'Processing failed');
    }
  };

  // Run full pipeline on dual files
  const handleProcessDual = async (doctorFile: File, patientFile: File) => {
    setProcessingStep('uploading');
    setProcessingError(null);
    setProcessingTime(undefined);

    try {
      setProcessingStep('transcribing');
      const res = await api.processEndToEnd(doctorFile, patientFile);
      setProcessingStep('reconstructing');
      setActiveTranscription(res.transcription);
      setActiveConversation(res.conversation);

      setProcessingStep('generating');
      setActiveNote(res.clinical_note);
      setActiveNoteId(res.note_id);
      setProcessingTime(res.processing_time_seconds);

      setProcessingStep('complete');
      await fetchSavedNotes();
    } catch (e: any) {
      setProcessingStep('error');
      setProcessingError(e?.response?.data?.detail || e.message || 'Processing failed');
    }
  };

  // Run PriMock57 Demo
  const handleRunDemo = async (day: number, consultation: number) => {
    setProcessingStep('reconstructing');
    setProcessingError(null);
    setProcessingTime(undefined);

    try {
      setProcessingStep('generating');
      const res = await api.runPriMockDemo(day, consultation);
      setActiveTranscription(res.transcription);
      setActiveConversation(res.conversation);
      setActiveNote(res.clinical_note);
      setActiveNoteId(res.note_id);
      setProcessingTime(res.processing_time_seconds);

      setProcessingStep('complete');
      await fetchSavedNotes();
    } catch (e: any) {
      setProcessingStep('error');
      setProcessingError(e?.response?.data?.detail || e.message || 'Demo processing failed');
    }
  };

  // Save reviewed note changes
  const handleSaveNote = async (updatedNote: ClinicalNote) => {
    if (!activeNoteId) return;
    try {
      const res = await api.updateNote(activeNoteId, updatedNote);
      setActiveNote(res.clinical_note);
      await fetchSavedNotes();
    } catch (e: any) {
      alert(`Failed to save note updates: ${e?.response?.data?.detail || e.message}`);
    }
  };

  // Select historical note
  const handleSelectHistoricalNote = (record: SavedNoteRecord) => {
    setActiveNote(record.clinical_note);
    setActiveNoteId(record.note_id);
    if (record.conversation) {
      // Re-create a minimal conversation view
      const lines = record.conversation.split('\n');
      const utterances = lines.map((l) => {
        const isDoc = l.toUpperCase().startsWith('DOCTOR:');
        return {
          speaker: isDoc ? 'Doctor' : 'Patient',
          text: l.replace(/^(DOCTOR|PATIENT):\s*/i, ''),
          start_time: 0,
          end_time: 0,
        };
      });
      setActiveConversation({
        consultation_id: record.consultation_id,
        utterances,
        formatted_conversation: record.conversation,
        total_utterances: utterances.length,
        doctor_utterances_count: utterances.filter((u) => u.speaker === 'Doctor').length,
        patient_utterances_count: utterances.filter((u) => u.speaker === 'Patient').length,
      });
    }
  };

  // Delete historical note
  const handleDeleteNote = async (noteId: string) => {
    try {
      await api.deleteNote(noteId);
      if (activeNoteId === noteId) {
        setActiveNote(null);
        setActiveNoteId(undefined);
      }
      await fetchSavedNotes();
    } catch (e: any) {
      alert(`Delete error: ${e?.response?.data?.detail || e.message}`);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Header */}
      <Header
        health={health}
        onRefreshHealth={fetchHealth}
        isLoadingModels={isLoadingModels}
        onLoadModels={handlePreloadModels}
      />

      {/* Main Container */}
      <main className="container" style={{ flex: 1, padding: '28px 24px' }}>
        
        {/* Top Control Bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
          <div>
            <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
              Consultation Speech-to-Note Pipeline
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>
              Upload audio tracks or select a benchmark consultation from PriMock57 to generate structured clinical notes.
            </p>
          </div>

          <button
            onClick={() => setShowHistory(!showHistory)}
            className="btn btn-secondary"
            style={{ fontSize: '13px' }}
          >
            <History size={16} />
            <span>{showHistory ? 'Hide Saved Notes' : `Saved Notes (${savedNotes.length})`}</span>
          </button>
        </div>

        {/* Layout Grid: If showHistory is open, 2 columns */}
        <div style={{ display: 'grid', gridTemplateColumns: showHistory ? '1fr 320px' : '1fr', gap: '24px' }}>
          
          {/* Left Column (Primary workflow) */}
          <div>
            
            {/* 1. Audio Upload */}
            <AudioUpload
              onProcessSingle={handleProcessSingle}
              onProcessDual={handleProcessDual}
              onRunDemo={handleRunDemo}
              isProcessing={processingStep !== 'idle' && processingStep !== 'complete' && processingStep !== 'error'}
            />

            {/* 2. Pipeline Processing Status */}
            <ProcessingStatus
              currentStep={processingStep}
              error={processingError}
              processingTime={processingTime}
            />

            {/* 3. Generated Clinical Note (Prominent result) */}
            {activeNote && (
              <ClinicalNoteEditor
                key={activeNoteId || 'editor'}
                initialNote={activeNote}
                noteId={activeNoteId}
                onSave={handleSaveNote}
              />
            )}

            {/* 4. Reconstructed Conversation View (Collapsible Accordion) */}
            {activeConversation && (
              <div className="card" style={{ marginBottom: '16px', padding: 0, overflow: 'hidden' }}>
                <button
                  type="button"
                  onClick={() => setShowConversation(!showConversation)}
                  style={{
                    width: '100%',
                    padding: '14px 18px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: '#f8fafc',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: 600,
                    color: 'var(--text-main)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <MessageSquare size={16} color="var(--primary)" />
                    <span>Reconstructed Dialogue ({activeConversation.total_utterances} turns)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
                    <span>{showConversation ? 'Hide dialogue' : 'Show dialogue'}</span>
                    {showConversation ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  </div>
                </button>
                {showConversation && (
                  <div style={{ padding: '16px 18px', borderTop: '1px solid var(--border)' }}>
                    <ConversationView conversation={activeConversation} />
                  </div>
                )}
              </div>
            )}

            {/* 5. Raw Transcription View (Collapsible Accordion) */}
            {activeTranscription && (
              <div className="card" style={{ marginBottom: '16px', padding: 0, overflow: 'hidden' }}>
                <button
                  type="button"
                  onClick={() => setShowTranscription(!showTranscription)}
                  style={{
                    width: '100%',
                    padding: '14px 18px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: '#f8fafc',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: 600,
                    color: 'var(--text-main)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FileAudio size={16} color="#059669" />
                    <span>Whisper Transcription ({activeTranscription.segments.length} segments)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
                    <span>{showTranscription ? 'Hide raw audio transcript' : 'Show raw audio transcript'}</span>
                    {showTranscription ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  </div>
                </button>
                {showTranscription && (
                  <div style={{ padding: '16px 18px', borderTop: '1px solid var(--border)' }}>
                    <TranscriptionView transcription={activeTranscription} />
                  </div>
                )}
              </div>
            )}

          </div>

          {/* Right Column (Saved notes sidebar) */}
          {showHistory && (
            <div>
              <SavedNotesList
                notes={savedNotes}
                onSelectNote={handleSelectHistoricalNote}
                onDeleteNote={handleDeleteNote}
                selectedNoteId={activeNoteId}
              />
            </div>
          )}

        </div>

      </main>

      {/* Footer */}
      <footer style={{ background: '#ffffff', borderTop: '1px solid var(--border)', padding: '16px 0', marginTop: 'auto' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
          <div>
            <strong>ClinicalNoteAI</strong> — Doctor–Patient LLM-Based Speech Recognition into Clinical Notes
          </div>
          <div>
            Academic Prototype • Apple MPS Accelerated • Whisper & Qwen2.5-1.5B LoRA
          </div>
        </div>
      </footer>

    </div>
  );
};

export default App;
