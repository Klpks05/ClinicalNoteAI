import React, { useState } from 'react';
import { Clock, Copy, Check, Users } from 'lucide-react';
import type { ReconstructResponse } from '../types';

interface ConversationViewProps {
  conversation: ReconstructResponse | null;
}

export const ConversationView: React.FC<ConversationViewProps> = ({ conversation }) => {
  const [copied, setCopied] = useState(false);

  if (!conversation) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(conversation.formatted_conversation);
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
            <Users size={18} color="var(--primary)" />
            <span>Reconstructed Doctor–Patient Dialogue</span>
          </h3>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '4px 0 0' }}>
            Chronological reconstruction ({conversation.doctor_utterances_count} Doctor turns, {conversation.patient_utterances_count} Patient turns)
          </p>
        </div>

        <button
          onClick={handleCopy}
          className="btn btn-secondary"
          style={{ fontSize: '12px', padding: '6px 12px' }}
        >
          {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
          <span>{copied ? 'Copied' : 'Copy Dialogue'}</span>
        </button>
      </div>

      {/* Utterances Stream */}
      <div style={{ maxHeight: '420px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px', paddingRight: '6px' }}>
        {conversation.utterances.map((u, i) => {
          const isDoctor = u.speaker.toLowerCase() === 'doctor';
          return (
            <div
              key={i}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: isDoctor ? 'flex-start' : 'flex-end',
              }}
            >
              <div
                style={{
                  maxWidth: '85%',
                  background: isDoctor ? '#eff6ff' : '#f0fdf4',
                  border: isDoctor ? '1px solid #bfdbfe' : '1px solid #bbf7d0',
                  borderRadius: '10px',
                  padding: '10px 14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '4px' }}>
                  <span
                    className={isDoctor ? 'badge badge-doctor' : 'badge badge-patient'}
                    style={{ fontSize: '10.5px', padding: '2px 8px' }}
                  >
                    {u.speaker}
                  </span>

                  {(u.start_time > 0 || u.end_time > 0) && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '10.5px', color: '#64748b' }}>
                      <Clock size={11} />
                      <span>{formatSeconds(u.start_time)} - {formatSeconds(u.end_time)}</span>
                    </div>
                  )}
                </div>

                <div style={{ fontSize: '13.5px', color: 'var(--text-main)', lineHeight: 1.45 }}>
                  {u.text}
                </div>
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
};
