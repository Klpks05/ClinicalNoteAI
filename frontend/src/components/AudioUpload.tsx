import React, { useState, useRef } from 'react';
import { Upload, FileAudio, Users, Play, Sparkles, Check, Info } from 'lucide-react';

interface AudioUploadProps {
  onProcessSingle: (file: File) => void;
  onProcessDual: (doctorFile: File, patientFile: File) => void;
  onRunDemo: (day: number, consultation: number) => void;
  isProcessing: boolean;
}

export const AudioUpload: React.FC<AudioUploadProps> = ({
  onProcessSingle,
  onProcessDual,
  onRunDemo,
  isProcessing,
}) => {
  const [mode, setMode] = useState<'single' | 'dual' | 'demo'>('single');
  
  // Single file state
  const [singleFile, setSingleFile] = useState<File | null>(null);
  
  // Dual file state
  const [doctorFile, setDoctorFile] = useState<File | null>(null);
  const [patientFile, setPatientFile] = useState<File | null>(null);

  // Demo selection state
  const [demoDay, setDemoDay] = useState<number>(1);
  const [demoConsultation, setDemoConsultation] = useState<number>(1);

  const singleInputRef = useRef<HTMLInputElement>(null);
  const doctorInputRef = useRef<HTMLInputElement>(null);
  const patientInputRef = useRef<HTMLInputElement>(null);

  const handleStartProcessing = () => {
    if (mode === 'single' && singleFile) {
      onProcessSingle(singleFile);
    } else if (mode === 'dual' && doctorFile && patientFile) {
      onProcessDual(doctorFile, patientFile);
    } else if (mode === 'demo') {
      onRunDemo(demoDay, demoConsultation);
    }
  };

  return (
    <div className="card" style={{ marginBottom: '24px' }}>
      
      {/* Mode Switcher Tabs */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border)', paddingBottom: '14px', marginBottom: '18px' }}>
        <button
          type="button"
          onClick={() => setMode('single')}
          className={`btn ${mode === 'single' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ fontSize: '13px', padding: '7px 14px' }}
        >
          <FileAudio size={15} />
          <span>Single Consultation Audio</span>
        </button>

        <button
          type="button"
          onClick={() => setMode('dual')}
          className={`btn ${mode === 'dual' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ fontSize: '13px', padding: '7px 14px' }}
        >
          <Users size={15} />
          <span>Dual Track (Doctor + Patient)</span>
        </button>

        <button
          type="button"
          onClick={() => setMode('demo')}
          className={`btn ${mode === 'demo' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ fontSize: '13px', padding: '7px 14px', background: mode === 'demo' ? '#8b5cf6' : undefined, color: mode === 'demo' ? '#fff' : undefined }}
        >
          <Sparkles size={15} />
          <span>PriMock57 Dataset Demo</span>
        </button>
      </div>

      {/* Mode 1: Single Audio Upload */}
      {mode === 'single' && (
        <div>
          <div
            onClick={() => singleInputRef.current?.click()}
            style={{
              border: '2px dashed var(--border-strong)',
              borderRadius: '10px',
              padding: '32px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              background: singleFile ? '#f0f9ff' : '#fafafa',
              borderColor: singleFile ? 'var(--primary)' : 'var(--border-strong)',
              transition: 'all 0.2s ease',
            }}
          >
            <input
              ref={singleInputRef}
              type="file"
              accept=".wav,.mp3,.m4a,.flac"
              style={{ display: 'none' }}
              onChange={(e) => {
                if (e.target.files?.[0]) setSingleFile(e.target.files[0]);
              }}
            />
            
            <div style={{
              width: '46px',
              height: '46px',
              borderRadius: '50%',
              background: singleFile ? 'var(--primary-light)' : '#e2e8f0',
              color: singleFile ? 'var(--primary)' : '#64748b',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 12px'
            }}>
              {singleFile ? <Check size={24} /> : <Upload size={22} />}
            </div>

            {singleFile ? (
              <div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--primary-hover)' }}>
                  {singleFile.name}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {(singleFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for speech recognition
                </div>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
                  Click or drag consultation recording here
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Supports WAV, MP3, M4A, FLAC (Max 100MB)
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Mode 2: Dual Track Doctor & Patient */}
      {mode === 'dual' && (
        <div>
          <div style={{
            background: 'var(--primary-light)',
            borderRadius: '8px',
            padding: '10px 14px',
            fontSize: '12px',
            color: '#0369a1',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '16px'
          }}>
            <Info size={16} style={{ flexShrink: 0 }} />
            <span>
              Upload separate tracks for Doctor and Patient (such as recordings from PriMock57). Both will be transcribed with timestamps and merged chronologically.
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            
            {/* Doctor Track */}
            <div
              onClick={() => doctorInputRef.current?.click()}
              style={{
                border: '2px dashed var(--border-strong)',
                borderRadius: '8px',
                padding: '24px 16px',
                textAlign: 'center',
                cursor: 'pointer',
                background: doctorFile ? '#f0f9ff' : '#fafafa',
                borderColor: doctorFile ? 'var(--primary)' : 'var(--border-strong)',
              }}
            >
              <input
                ref={doctorInputRef}
                type="file"
                accept=".wav,.mp3,.m4a,.flac"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files?.[0]) setDoctorFile(e.target.files[0]);
                }}
              />
              <span className="badge badge-doctor" style={{ marginBottom: '8px' }}>Doctor Track</span>
              <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-main)', marginTop: '6px' }}>
                {doctorFile ? doctorFile.name : 'Select Doctor Audio'}
              </div>
            </div>

            {/* Patient Track */}
            <div
              onClick={() => patientInputRef.current?.click()}
              style={{
                border: '2px dashed var(--border-strong)',
                borderRadius: '8px',
                padding: '24px 16px',
                textAlign: 'center',
                cursor: 'pointer',
                background: patientFile ? '#f0fdf4' : '#fafafa',
                borderColor: patientFile ? '#22c55e' : 'var(--border-strong)',
              }}
            >
              <input
                ref={patientInputRef}
                type="file"
                accept=".wav,.mp3,.m4a,.flac"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files?.[0]) setPatientFile(e.target.files[0]);
                }}
              />
              <span className="badge badge-patient" style={{ marginBottom: '8px' }}>Patient Track</span>
              <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-main)', marginTop: '6px' }}>
                {patientFile ? patientFile.name : 'Select Patient Audio'}
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Mode 3: PriMock57 Dataset Demo */}
      {mode === 'demo' && (
        <div style={{ background: '#f5f3ff', border: '1px solid #ddd6fe', borderRadius: '10px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#6d28d9', fontWeight: 600, fontSize: '14px', marginBottom: '8px' }}>
            <Sparkles size={18} />
            <span>Instant PriMock57 Consultation Demo</span>
          </div>
          <p style={{ fontSize: '12.5px', color: '#5b21b6', marginBottom: '16px' }}>
            Loads actual consultations directly from <code>data/primock57/</code>, reconstructs chronological dialogue from TextGrids, and runs the Fine-Tuned Qwen LoRA adapter to generate a full clinical note.
          </p>

          <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
            <div>
              <label style={{ fontSize: '12px', fontWeight: 600, color: '#4c1d95', display: 'block', marginBottom: '4px' }}>
                Consultation Day:
              </label>
              <select
                value={demoDay}
                onChange={(e) => setDemoDay(Number(e.target.value))}
                style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid #c4b5fd', fontSize: '13px', background: '#fff' }}
              >
                {[1, 2, 3, 4, 5].map((d) => (
                  <option key={d} value={d}>Day {d}</option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ fontSize: '12px', fontWeight: 600, color: '#4c1d95', display: 'block', marginBottom: '4px' }}>
                Consultation ID:
              </label>
              <select
                value={demoConsultation}
                onChange={(e) => setDemoConsultation(Number(e.target.value))}
                style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid #c4b5fd', fontSize: '13px', background: '#fff' }}
              >
                {Array.from({ length: 12 }, (_, i) => i + 1).map((c) => (
                  <option key={c} value={c}>Consultation {c < 10 ? `0${c}` : c}</option>
                ))}
              </select>
            </div>

            <div style={{ fontSize: '12px', color: '#6d28d9', alignSelf: 'flex-end', paddingBottom: '8px' }}>
              Selected: <strong>day{demoDay}_consultation{demoConsultation < 10 ? `0${demoConsultation}` : demoConsultation}</strong>
            </div>
          </div>
        </div>
      )}

      {/* Action Footer */}
      <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
        <button
          type="button"
          onClick={handleStartProcessing}
          disabled={
            isProcessing ||
            (mode === 'single' && !singleFile) ||
            (mode === 'dual' && (!doctorFile || !patientFile))
          }
          className="btn btn-primary"
          style={{ padding: '12px 24px', fontSize: '14px' }}
        >
          <Play size={16} />
          <span>
            {isProcessing
              ? 'Analyzing Consultation...'
              : mode === 'demo'
              ? 'Run PriMock57 Consultation Demo'
              : 'Generate Structured Clinical Note'}
          </span>
        </button>
      </div>

    </div>
  );
};
