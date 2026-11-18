"""
Conversation reconstruction service.

Parses TextGrid transcripts, merges doctor and patient utterances
chronologically, and formats the conversation.

Reuses logic from:
- scripts/end_to_end_test.py (TextGrid parsing, tag cleaning)
- scripts/prepare_primock57.py (TextGrid reading, chronological merge)
- data/primock57/scripts/textgrid_to_transcript.py (combined transcript)
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """A single utterance in a conversation."""

    speaker: str  # "Doctor" or "Patient"
    text: str
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class Conversation:
    """A reconstructed doctor-patient conversation."""

    utterances: list[Utterance] = field(default_factory=list)
    consultation_id: str | None = None

    @property
    def doctor_utterances(self) -> list[Utterance]:
        return [u for u in self.utterances if u.speaker == "Doctor"]

    @property
    def patient_utterances(self) -> list[Utterance]:
        return [u for u in self.utterances if u.speaker == "Patient"]

    def to_formatted_text(self) -> str:
        """Format as 'DOCTOR: ...\nPATIENT: ...' text."""
        lines = []
        for u in self.utterances:
            lines.append(f"{u.speaker.upper()}: {u.text}")
        return "\n".join(lines)


class ConversationServiceError(Exception):
    """Raised when conversation reconstruction fails."""
    pass


class ConversationService:
    """
    Service for reconstructing doctor-patient conversations
    from various input formats.
    """

    # TextGrid annotation tags to clean
    _TAG_PATTERNS = [
        (re.compile(r"<UNSURE>(.*?)</UNSURE>", re.DOTALL), r"\1"),
        (re.compile(r"<UNIN/>"), ""),
        (re.compile(r"<INAUDIBLE_SPEECH/>"), ""),
    ]

    def clean_text(self, text: str) -> str:
        """
        Remove TextGrid annotation tags and normalize whitespace.

        Handles: <UNSURE>...</UNSURE>, <UNIN/>, <INAUDIBLE_SPEECH/>

        Args:
            text: Raw transcript text with possible annotations.

        Returns:
            Cleaned text string.
        """
        for pattern, replacement in self._TAG_PATTERNS:
            text = pattern.sub(replacement, text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def parse_textgrid(
        self,
        file_path: str | Path,
        speaker: str,
    ) -> list[Utterance]:
        """
        Parse a Praat TextGrid file and extract utterances.

        Uses regex-based parsing matching the approach in
        scripts/end_to_end_test.py.

        Args:
            file_path: Path to the TextGrid file.
            speaker: Speaker label ("Doctor" or "Patient").

        Returns:
            List of Utterance objects with timestamps.

        Raises:
            ConversationServiceError: If parsing fails.
        """
        path = Path(file_path)

        if not path.exists():
            raise ConversationServiceError(
                f"TextGrid file not found: {path}"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise ConversationServiceError(
                f"Failed to read TextGrid file: {e}"
            ) from e

        # Regex pattern from end_to_end_test.py
        pattern = (
            r"xmin\s*=\s*([0-9.]+)\s*"
            r"xmax\s*=\s*([0-9.]+)\s*"
            r"text\s*=\s*\"(.*?)\""
        )

        matches = re.findall(pattern, content, flags=re.DOTALL)

        utterances = []

        for start_str, end_str, text in matches:
            text = text.strip()

            if not text:
                continue

            # Clean annotation tags
            text = self.clean_text(text)

            if not text:
                continue

            utterances.append(
                Utterance(
                    speaker=speaker,
                    text=text,
                    start_time=float(start_str),
                    end_time=float(end_str),
                )
            )

        logger.info(
            f"Parsed {len(utterances)} utterances from "
            f"{path.name} ({speaker})"
        )

        return utterances

    def reconstruct_from_textgrids(
        self,
        doctor_textgrid: str | Path,
        patient_textgrid: str | Path,
        consultation_id: str | None = None,
    ) -> Conversation:
        """
        Reconstruct a conversation from separate doctor and patient
        TextGrid files.

        Merges both speakers' utterances chronologically by timestamp.

        Args:
            doctor_textgrid: Path to the doctor's TextGrid file.
            patient_textgrid: Path to the patient's TextGrid file.
            consultation_id: Optional consultation identifier.

        Returns:
            Conversation object with chronologically sorted utterances.
        """
        doctor_utterances = self.parse_textgrid(
            doctor_textgrid, "Doctor"
        )
        patient_utterances = self.parse_textgrid(
            patient_textgrid, "Patient"
        )

        # Merge and sort chronologically
        all_utterances = doctor_utterances + patient_utterances
        all_utterances.sort(key=lambda u: u.start_time)

        conversation = Conversation(
            utterances=all_utterances,
            consultation_id=consultation_id,
        )

        logger.info(
            f"Reconstructed conversation: "
            f"{len(doctor_utterances)} doctor + "
            f"{len(patient_utterances)} patient = "
            f"{len(all_utterances)} total utterances"
        )

        return conversation

    def reconstruct_from_dual_transcripts(
        self,
        doctor_text: str,
        patient_text: str,
        doctor_segments: list[dict] | None = None,
        patient_segments: list[dict] | None = None,
        consultation_id: str | None = None,
    ) -> Conversation:
        """
        Reconstruct a conversation from separate doctor and patient
        transcription texts (e.g., from Whisper on dual audio tracks).

        If timestamped segments are provided, utterances are merged
        chronologically. Otherwise, they are interleaved.

        Args:
            doctor_text: Full doctor transcription text.
            patient_text: Full patient transcription text.
            doctor_segments: Optional timestamped segments for doctor.
            patient_segments: Optional timestamped segments for patient.
            consultation_id: Optional consultation identifier.

        Returns:
            Conversation object.
        """
        utterances = []

        if doctor_segments and patient_segments:
            # Use timestamps for chronological merge
            for seg in doctor_segments:
                text = self.clean_text(seg.get("text", "").strip())
                if text:
                    utterances.append(
                        Utterance(
                            speaker=seg.get("speaker") or "Doctor",
                            text=text,
                            start_time=seg.get("start", 0.0),
                            end_time=seg.get("end", 0.0),
                        )
                    )

            for seg in patient_segments:
                text = self.clean_text(seg.get("text", "").strip())
                if text:
                    utterances.append(
                        Utterance(
                            speaker=seg.get("speaker") or "Patient",
                            text=text,
                            start_time=seg.get("start", 0.0),
                            end_time=seg.get("end", 0.0),
                        )
                    )

            utterances.sort(key=lambda u: u.start_time)
        else:
            # No timestamps — split by sentences and interleave
            doctor_sentences = [
                s.strip() for s in doctor_text.split(".")
                if s.strip()
            ]
            patient_sentences = [
                s.strip() for s in patient_text.split(".")
                if s.strip()
            ]

            for sent in doctor_sentences:
                utterances.append(
                    Utterance(speaker="Doctor", text=sent)
                )

            for sent in patient_sentences:
                utterances.append(
                    Utterance(speaker="Patient", text=sent)
                )

        return Conversation(
            utterances=utterances,
            consultation_id=consultation_id,
        )

    def diarize_and_reconstruct_single_audio(
        self,
        text: str,
        segments: list[dict] | None = None,
        audio_path: str | Path | None = None,
        consultation_id: str | None = None,
    ) -> Conversation:
        """
        Reconstruct a conversation from a single audio transcription.
        If segments and audio_path are available, uses MFCC feature clustering
        to separate Doctor and Patient speakers.
        """
        if not segments:
            return self.reconstruct_from_single_transcript(text, consultation_id)

        # Check if text already has DOCTOR/PATIENT labels
        if bool(re.search(r"(?i)(DOCTOR|PATIENT)\s*:", text)):
            return self.reconstruct_from_single_transcript(text, consultation_id)

        utterances = []
        labels = None

        if audio_path and Path(audio_path).exists() and len(segments) >= 2:
            try:
                import numpy as np
                import librosa
                from sklearn.cluster import AgglomerativeClustering

                # Load audio up to 10 minutes for fast MFCC extraction
                y, sr = librosa.load(str(audio_path), sr=16000, duration=600)
                features = []
                valid_indices = []

                for idx, seg in enumerate(segments):
                    st = int(seg.get("start", 0.0) * sr)
                    en = int(seg.get("end", 0.0) * sr)
                    if en - st > sr * 0.3 and st < len(y):  # At least 300ms of audio
                        seg_y = y[st:min(en, len(y))]
                        mfcc = librosa.feature.mfcc(y=seg_y, sr=sr, n_mfcc=13)
                        feat = np.mean(mfcc, axis=1)
                        features.append(feat)
                        valid_indices.append(idx)

                if len(features) >= 2:
                    clustering = AgglomerativeClustering(n_clusters=2).fit(features)
                    # First speaker cluster is mapped to Doctor, other to Patient
                    first_cluster = clustering.labels_[0]
                    labels = {}
                    for v_idx, cluster_id in zip(valid_indices, clustering.labels_):
                        labels[v_idx] = "Doctor" if cluster_id == first_cluster else "Patient"
            except Exception as e:
                logger.warning(f"Acoustic speaker clustering failed: {e}. Falling back to alternating turns.")

        # Assign speaker labels
        current_speaker = "Doctor"
        for idx, seg in enumerate(segments):
            seg_text = self.clean_text(seg.get("text", "").strip())
            if not seg_text:
                continue

            if seg.get("speaker"):
                speaker = seg["speaker"]
            elif labels and idx in labels:
                speaker = labels[idx]
            else:
                speaker = current_speaker
                if seg_text.endswith("?"):
                    current_speaker = "Patient"
                elif current_speaker == "Patient" and (len(seg_text.split()) > 10 or seg_text.endswith(".")):
                    current_speaker = "Doctor"

            utterances.append(
                Utterance(
                    speaker=speaker,
                    text=seg_text,
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                )
            )

        if not utterances:
            return self.reconstruct_from_single_transcript(text, consultation_id)

        return Conversation(utterances=utterances, consultation_id=consultation_id)

    def reconstruct_from_single_transcript(
        self,
        text: str,
        consultation_id: str | None = None,
        segments: list[dict] | None = None,
        audio_path: str | Path | None = None,
    ) -> Conversation:
        """
        Reconstruct a conversation from a single mixed transcript.

        If segments are provided and no labels exist, delegates to diarization.
        If the text contains 'DOCTOR:' and 'PATIENT:' labels, those are used.
        Otherwise, the entire text is treated as a single speaker block.
        """
        if segments and len(segments) > 1 and not re.search(r"(?i)(DOCTOR|PATIENT)\s*:", text):
            return self.diarize_and_reconstruct_single_audio(
                text=text,
                segments=segments,
                audio_path=audio_path,
                consultation_id=consultation_id,
            )

        utterances = []

        # Check if text has speaker labels
        has_labels = bool(
            re.search(r"(?i)(DOCTOR|PATIENT)\s*:", text)
        )

        if has_labels:
            # Split by speaker labels
            pattern = re.compile(
                r"(?i)(DOCTOR|PATIENT)\s*:\s*(.*?)(?=(?:DOCTOR|PATIENT)\s*:|$)",
                re.DOTALL,
            )

            for match in pattern.finditer(text):
                speaker_raw = match.group(1).strip()
                content = match.group(2).strip()

                if not content:
                    continue

                speaker = (
                    "Doctor" if speaker_raw.upper() == "DOCTOR"
                    else "Patient"
                )

                utterances.append(
                    Utterance(speaker=speaker, text=content)
                )
        else:
            # No labels — treat as single block transcript
            if text.strip():
                utterances.append(
                    Utterance(
                        speaker="Unknown",
                        text=text.strip(),
                    )
                )

        return Conversation(
            utterances=utterances,
            consultation_id=consultation_id,
        )
