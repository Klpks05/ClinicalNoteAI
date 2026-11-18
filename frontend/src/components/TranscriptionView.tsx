import React, { useState } from 'react';
import { Mic, Clock, Copy, Check } from 'lucide-react';
import type { TranscribeResponse } from '../types';

interface TranscriptionViewProps {
  transcription: TranscribeResponse | null;
}

export const TranscriptionView: React.FC<TranscriptionViewProps> = ({ transcription }) => {
  const [showSegments, setShowSegments] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!transcription) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(transcription.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatSeconds = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${mins}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="card" style={{ marginBottom: '24px' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '14px', marginBottom: '16px' }}>
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '600', color: 'var(--text-main)', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Mic size={18} color="var(--primary)" />
            <span>Whisper Speech Transcription</span>
          </h3>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '4px 0 0' }}>
            Decoded audio text ({transcription.segments?.length || 0} timestamped chunks)
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setShowSegments(!showSegments)}
            className="btn btn-secondary"
            style={{ fontSize: '12px', padding: '6px 12px' }}
          >
            <Clock size={13} />
            <span>{showSegments ? 'Show Full Text' : 'Show Timestamps'}</span>
          </button>

          <button
            onClick={handleCopy}
            className="btn btn-secondary"
            style={{ fontSize: '12px', padding: '6px 12px' }}
          >
            {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      {/* Content */}
      {showSegments && transcription.segments && transcription.segments.length > 0 ? (
        <div style={{ maxHeight: '350px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {transcription.segments.map((seg, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: '12px',
                padding: '6px 10px',
                background: 'var(--bg-subtle)',
                borderRadius: '6px',
                fontSize: '13px',
              }}
            >
              <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--primary)', flexShrink: 0 }}>
                [{formatSeconds(seg.start)} - {formatSeconds(seg.end)}]
              </span>
              <span style={{ color: 'var(--text-main)' }}>{seg.text}</span>
            </div>
          ))}
        </div>
      ) : (
        <div
          style={{
            fontSize: '13.5px',
            color: 'var(--text-main)',
            lineHeight: '1.6',
            whiteSpace: 'pre-wrap',
            background: 'var(--bg-subtle)',
            padding: '14px',
            borderRadius: '8px',
            maxHeight: '300px',
            overflowY: 'auto',
          }}
        >
          {transcription.text || 'No transcription text.'}
        </div>
      )}

    </div>
  );
};
