import React from 'react';
import { FileText, Calendar, Trash2, ArrowRight } from 'lucide-react';
import type { SavedNoteRecord } from '../types';

interface SavedNotesListProps {
  notes: SavedNoteRecord[];
  onSelectNote: (note: SavedNoteRecord) => void;
  onDeleteNote: (noteId: string) => void;
  selectedNoteId?: string;
}

export const SavedNotesList: React.FC<SavedNotesListProps> = ({
  notes,
  onSelectNote,
  onDeleteNote,
  selectedNoteId,
}) => {
  if (notes.length === 0) {
    return (
      <div className="card" style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <FileText size={24} style={{ margin: '0 auto 8px', color: '#94a3b8' }} />
        <div style={{ fontSize: '13px', fontWeight: 500 }}>No saved clinical notes yet</div>
        <p style={{ fontSize: '11.5px', marginTop: '4px' }}>
          Process consultation audio or run a PriMock57 demo to generate and save notes.
        </p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: '16px' }}>
      <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '12px' }}>
        Recent Clinical Notes ({notes.length})
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '360px', overflowY: 'auto' }}>
        {notes.map((item) => {
          const isSelected = selectedNoteId === item.note_id;
          return (
            <div
              key={item.note_id}
              onClick={() => onSelectNote(item)}
              style={{
                border: isSelected ? '1px solid var(--primary)' : '1px solid var(--border)',
                background: isSelected ? 'var(--primary-light)' : '#ffffff',
                borderRadius: '8px',
                padding: '10px 12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '8px',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.clinical_note?.chief_complaint && !item.clinical_note.chief_complaint.includes('Not mentioned')
                    ? item.clinical_note.chief_complaint
                    : item.filename || `Note #${item.note_id.slice(0, 8)}`}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Calendar size={11} />
                  <span>{new Date(item.created_at).toLocaleDateString()} {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm('Delete this saved clinical note?')) {
                      onDeleteNote(item.note_id);
                    }
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#ef4444',
                    cursor: 'pointer',
                    padding: '4px',
                    borderRadius: '4px',
                  }}
                  title="Delete note"
                >
                  <Trash2 size={14} />
                </button>
                <ArrowRight size={14} color={isSelected ? 'var(--primary)' : '#94a3b8'} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
