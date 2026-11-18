import React, { useState } from 'react';
import {
  FileText,
  Save,
  Download,
  Check,
  Edit3,
  Calendar,
  Sparkles,
  RotateCcw
} from 'lucide-react';
import type { ClinicalNote } from '../types';
import { DraftBanner } from './DraftBanner';

interface ClinicalNoteEditorProps {
  initialNote: ClinicalNote;
  noteId?: string;
  onSave?: (updatedNote: ClinicalNote) => Promise<void>;
}

interface SectionConfig {
  key: keyof Omit<ClinicalNote, 'metadata'>;
  title: string;
  category: 'Subjective' | 'Objective' | 'Assessment' | 'Plan';
}

const SECTIONS: SectionConfig[] = [
  { key: 'chief_complaint', title: 'Chief Complaint', category: 'Subjective' },
  { key: 'history_of_present_illness', title: 'History of Present Illness (HPI)', category: 'Subjective' },
  { key: 'past_medical_history', title: 'Relevant Medical History', category: 'Subjective' },
  { key: 'medications', title: 'Current Medications', category: 'Subjective' },
  { key: 'allergies', title: 'Allergies', category: 'Subjective' },
  { key: 'review_of_systems', title: 'Review of Systems (ROS)', category: 'Subjective' },
  { key: 'examination', title: 'Physical Examination', category: 'Objective' },
  { key: 'assessment', title: 'Clinical Assessment', category: 'Assessment' },
  { key: 'plan', title: 'Plan and Follow-up', category: 'Plan' },
];

export const ClinicalNoteEditor: React.FC<ClinicalNoteEditorProps> = ({
  initialNote,
  noteId,
  onSave,
}) => {
  const [note, setNote] = useState<ClinicalNote>(initialNote);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleFieldChange = (key: keyof Omit<ClinicalNote, 'metadata'>, value: string) => {
    setNote((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleSaveAll = async () => {
    if (!onSave) return;
    setIsSaving(true);
    try {
      await onSave(note);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2500);
      setEditingKey(null);
    } finally {
      setIsSaving(false);
    }
  };

  const handleExportText = () => {
    let content = `============================================================\n`;
    content += `CLINICAL CONSULTATION NOTE\n`;
    content += `(DRAFT — Requires Clinician Review and Verification)\n`;
    content += `============================================================\n\n`;
    SECTIONS.forEach((sec) => {
      content += `${sec.title.toUpperCase()}\n`;
      content += `${'-'.repeat(sec.title.length)}\n`;
      content += `${note[sec.key] || 'Not documented.'}\n\n`;
    });
    content += `============================================================\n`;
    content += `Generated: ${note.metadata.generated_at}\n`;
    content += `Model: ${note.metadata.model_name}\n`;
    content += `Adapter: ${note.metadata.adapter_name}\n`;
    content += `Notice: ${note.metadata.disclaimer}\n`;
    content += `============================================================\n`;

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `clinical_note_${noteId || 'draft'}.txt`;
    link.click();
  };

  const handleExportJSON = () => {
    const blob = new Blob([JSON.stringify(note, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `clinical_note_${noteId || 'draft'}.json`;
    link.click();
  };

  const getCategoryColor = (cat: SectionConfig['category']) => {
    switch (cat) {
      case 'Subjective': return '#0284c7';
      case 'Objective': return '#059669';
      case 'Assessment': return '#d97706';
      case 'Plan': return '#7c3aed';
    }
  };

  return (
    <div className="card" style={{ marginBottom: '24px' }}>
      
      {/* Draft Disclaimer Banner */}
      <DraftBanner />

      {/* Action Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '16px', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={20} color="var(--primary)" />
            <span>Structured Clinical Consultation Note</span>
          </h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Sparkles size={12} color="#0284c7" />
              <span>LoRA Fine-Tuned (MTS-Dialog)</span>
            </span>
            <span>•</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Calendar size={12} />
              <span>{new Date(note.metadata.generated_at).toLocaleDateString()} {new Date(note.metadata.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
            </span>
          </div>
        </div>

        {/* Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={handleExportText}
            className="btn btn-secondary"
            style={{ fontSize: '12.5px', padding: '7px 12px' }}
            title="Download plain text file"
          >
            <Download size={14} />
            <span>Export Text</span>
          </button>

          <button
            onClick={handleExportJSON}
            className="btn btn-secondary"
            style={{ fontSize: '12.5px', padding: '7px 12px' }}
            title="Download structured JSON"
          >
            <Download size={14} />
            <span>Export JSON</span>
          </button>

          {onSave && (
            <button
              onClick={handleSaveAll}
              disabled={isSaving}
              className="btn btn-primary"
              style={{ fontSize: '12.5px', padding: '7px 16px' }}
            >
              {saveSuccess ? (
                <>
                  <Check size={14} />
                  <span>Saved!</span>
                </>
              ) : isSaving ? (
                <span>Saving...</span>
              ) : (
                <>
                  <Save size={14} />
                  <span>Save Review</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Sections List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {SECTIONS.map((sec) => {
          const value = (note[sec.key] as string) || '';
          const isNotMentioned = !value || value.toLowerCase().includes('not mentioned') || value.toLowerCase().includes('not documented');
          const isEditing = editingKey === sec.key;

          return (
            <div
              key={sec.key}
              style={{
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '14px 18px',
                background: isNotMentioned ? '#fafafa' : '#ffffff',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    style={{
                      width: '6px',
                      height: '16px',
                      borderRadius: '3px',
                      background: getCategoryColor(sec.category),
                      display: 'inline-block',
                    }}
                  />
                  <h4 style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-main)', margin: 0 }}>
                    {sec.title}
                  </h4>
                  <span style={{ fontSize: '10.5px', color: 'var(--text-muted)', background: 'var(--bg-subtle)', padding: '2px 6px', borderRadius: '4px' }}>
                    {sec.category}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-light)' }}>
                    {value.length} chars
                  </span>
                  <button
                    type="button"
                    onClick={() => setEditingKey(isEditing ? null : sec.key)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: isEditing ? 'var(--primary)' : 'var(--text-muted)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      fontSize: '11.5px',
                    }}
                  >
                    <Edit3 size={13} />
                    <span>{isEditing ? 'Collapse' : 'Edit'}</span>
                  </button>
                </div>
              </div>

              {isEditing ? (
                <div>
                  <textarea
                    rows={4}
                    value={value}
                    onChange={(e) => handleFieldChange(sec.key, e.target.value)}
                    className="textarea-input"
                    placeholder={`Enter ${sec.title}...`}
                  />
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '6px' }}>
                    <button
                      type="button"
                      onClick={() => handleFieldChange(sec.key, 'Not documented.')}
                      className="btn btn-secondary"
                      style={{ fontSize: '11px', padding: '4px 8px' }}
                    >
                      <RotateCcw size={11} />
                      <span>Set "Not documented"</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setEditingKey(null)}
                      className="btn btn-primary"
                      style={{ fontSize: '11px', padding: '4px 10px' }}
                    >
                      <Check size={11} />
                      <span>Done</span>
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  style={{
                    fontSize: '13.5px',
                    color: isNotMentioned ? '#94a3b8' : 'var(--text-main)',
                    fontStyle: isNotMentioned ? 'italic' : 'normal',
                    whiteSpace: 'pre-wrap',
                    lineHeight: '1.5',
                  }}
                >
                  {value || 'Not documented.'}
                </div>
              )}
            </div>
          );
        })}
      </div>

    </div>
  );
};
