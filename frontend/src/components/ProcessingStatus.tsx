import React from 'react';
import { CheckCircle2, Loader2, Circle, Clock } from 'lucide-react';

export type ProcessingStep = 'idle' | 'uploading' | 'transcribing' | 'reconstructing' | 'generating' | 'complete' | 'error';

interface ProcessingStatusProps {
  currentStep: ProcessingStep;
  error?: string | null;
  processingTime?: number;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  currentStep,
  error,
  processingTime,
}) => {
  if (currentStep === 'idle') return null;

  const steps = [
    { id: 'uploading', label: 'Uploading consultation audio', desc: 'Validating format and audio characteristics' },
    { id: 'transcribing', label: 'Whisper Speech Recognition', desc: 'Decoding speech to text with timestamps' },
    { id: 'reconstructing', label: 'Doctor–Patient Reconstruction', desc: 'Chronologically aligning speaker turns' },
    { id: 'generating', label: 'Clinical Note Generation', desc: 'Fine-tuned Qwen LoRA extracting structured fields on Apple MPS (~35–45s)' },
  ];

  const getStepIndex = (step: ProcessingStep) => {
    switch (step) {
      case 'uploading': return 0;
      case 'transcribing': return 1;
      case 'reconstructing': return 2;
      case 'generating': return 3;
      case 'complete': return 4;
      default: return -1;
    }
  };

  const currentIndex = getStepIndex(currentStep);

  return (
    <div className="card" style={{ marginBottom: '24px', background: '#f8fafc' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-main)', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>AI Pipeline Processing Status</span>
          {currentStep !== 'complete' && currentStep !== 'error' && (
            <span className="badge" style={{ background: '#e0f2fe', color: '#0369a1' }}>Active</span>
          )}
          {currentStep === 'complete' && (
            <span className="badge" style={{ background: '#dcfce7', color: '#15803d' }}>
              Completed in {processingTime ? `${processingTime}s` : 'done'}
            </span>
          )}
        </h3>

        {currentStep === 'complete' && processingTime && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
            <Clock size={14} />
            <span>Total execution: {processingTime}s</span>
          </div>
        )}
      </div>

      {error && (
        <div style={{
          background: '#fee2e2',
          border: '1px solid #fca5a5',
          borderRadius: '8px',
          padding: '12px',
          color: '#b91c1c',
          fontSize: '13px',
          marginBottom: '16px',
        }}>
          <strong>Processing Error:</strong> {error}
        </div>
      )}

      {/* Steps List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {steps.map((step, idx) => {
          const isDone = currentIndex > idx;
          const isActive = currentIndex === idx;
          const isPending = currentIndex < idx;

          return (
            <div
              key={step.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '10px 14px',
                borderRadius: '8px',
                background: isActive ? '#ffffff' : 'transparent',
                border: isActive ? '1px solid var(--primary-border)' : '1px solid transparent',
                boxShadow: isActive ? '0 1px 3px rgba(0,0,0,0.05)' : 'none',
                opacity: isPending ? 0.45 : 1,
                transition: 'all 0.2s ease',
              }}
            >
              {/* Icon */}
              <div style={{ flexShrink: 0 }}>
                {isDone && <CheckCircle2 size={18} color="#10b981" />}
                {isActive && <Loader2 size={18} color="var(--primary)" className="spinner" />}
                {isPending && <Circle size={18} color="#94a3b8" />}
              </div>

              {/* Step info */}
              <div style={{ flex: 1 }}>
                <div style={{
                  fontSize: '13.5px',
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? 'var(--primary-hover)' : 'var(--text-main)',
                }}>
                  {step.label}
                </div>
                <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                  {step.desc}
                </div>
              </div>

              {/* Status pill */}
              <div style={{ fontSize: '11px', fontWeight: 600 }}>
                {isDone && <span style={{ color: '#10b981' }}>Done</span>}
                {isActive && <span style={{ color: 'var(--primary)' }}>Running...</span>}
                {isPending && <span style={{ color: '#94a3b8' }}>Pending</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
