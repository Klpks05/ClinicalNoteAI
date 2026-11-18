import React from 'react';
import { Activity, Stethoscope, Cpu, RefreshCw } from 'lucide-react';
import type { HealthResponse } from '../types';

interface HeaderProps {
  health: HealthResponse | null;
  onRefreshHealth: () => void;
  isLoadingModels: boolean;
  onLoadModels: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  onRefreshHealth,
  isLoadingModels,
  onLoadModels,
}) => {
  return (
    <header style={{ background: '#ffffff', borderBottom: '1px solid var(--border)', padding: '16px 0' }}>
      <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
            color: '#fff',
            width: '42px',
            height: '42px',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 6px -1px rgba(2, 132, 199, 0.3)'
          }}>
            <Stethoscope size={24} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-main)', letterSpacing: '-0.3px', margin: 0 }}>
                ClinicalNoteAI
              </h1>
              <span style={{
                background: 'var(--primary-light)',
                color: 'var(--primary-hover)',
                fontSize: '11px',
                fontWeight: '600',
                padding: '2px 8px',
                borderRadius: '6px'
              }}>
                Prototype v1.0
              </span>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: 0 }}>
              Doctor–Patient Consultation Speech Recognition & Structured Clinical Notes
            </p>
          </div>
        </div>

        {/* Status badges & controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          
          {/* Whisper Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px',
            background: 'var(--bg-subtle)',
            padding: '6px 12px',
            borderRadius: '8px',
            border: '1px solid var(--border)'
          }}>
            <Activity size={14} color={health?.whisper_model_loaded ? '#10b981' : '#f59e0b'} />
            <span style={{ fontWeight: 500 }}>Whisper:</span>
            <span style={{ color: health?.whisper_model_loaded ? '#10b981' : '#64748b', fontWeight: 600 }}>
              {health?.whisper_model_loaded ? 'Loaded' : 'Ready (on-demand)'}
            </span>
          </div>

          {/* Qwen LoRA Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px',
            background: 'var(--bg-subtle)',
            padding: '6px 12px',
            borderRadius: '8px',
            border: '1px solid var(--border)'
          }}>
            <Cpu size={14} color={health?.clinical_llm_loaded ? '#10b981' : '#0284c7'} />
            <span style={{ fontWeight: 500 }}>Qwen LoRA:</span>
            <span style={{ color: health?.clinical_llm_loaded ? '#10b981' : '#64748b', fontWeight: 600 }}>
              {health?.clinical_llm_loaded ? 'In Memory' : 'MTS Adapter'}
            </span>
          </div>

          {/* Preload button if not loaded */}
          {!health?.clinical_llm_loaded && (
            <button
              onClick={onLoadModels}
              disabled={isLoadingModels}
              className="btn btn-secondary"
              style={{ fontSize: '12px', padding: '6px 12px' }}
              title="Preload models into GPU/MPS memory for faster inference"
            >
              {isLoadingModels ? (
                <>
                  <Activity size={14} className="spinner" />
                  <span>Loading Models...</span>
                </>
              ) : (
                <>
                  <Cpu size={14} />
                  <span>Preload Models</span>
                </>
              )}
            </button>
          )}

          {/* Refresh Health Button */}
          <button
            onClick={onRefreshHealth}
            className="btn btn-secondary"
            style={{ fontSize: '12px', padding: '6px 8px' }}
            title="Refresh backend status"
          >
            <RefreshCw size={14} />
          </button>
        </div>

      </div>
    </header>
  );
};
