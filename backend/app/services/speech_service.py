"""
Speech recognition service using OpenAI Whisper.

Provides audio transcription with timestamp support.
Reuses logic from scripts/test_whisper.py.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


from typing import Any

@dataclass
class TimestampedSegment:
    """A segment of transcribed speech with timestamps."""

    text: str
    start: float
    end: float
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert segment to dictionary format."""
        data: dict[str, Any] = {
            "text": self.text,
            "start": float(self.start),
            "end": float(self.end),
        }
        if self.speaker is not None:
            data["speaker"] = self.speaker
        return data

    def model_dump(self) -> dict[str, Any]:
        """Compatibility alias for model_dump."""
        return self.to_dict()


@dataclass
class TranscriptionResult:
    """Complete transcription result from Whisper."""

    text: str
    segments: list[TimestampedSegment] = field(default_factory=list)
    language: str = "en"
    duration_seconds: float | None = None


class SpeechServiceError(Exception):
    """Raised when speech recognition fails."""
    pass


class SpeechService:
    """
    Whisper-based speech recognition service.

    Supports Apple MPS acceleration with CPU fallback.
    Uses singleton pattern for model management.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str = "openai/whisper-small.en",
    ):
        if self._initialized:
            return

        self.model_name = model_name
        self._pipeline = None
        self._device = None
        self._initialized = True

    @property
    def device(self) -> str:
        """Get the compute device (mps or cpu)."""
        if self._device is None:
            # Whisper has known issues with MPS on some operations,
            # so we default to CPU for reliability
            if torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            logger.info(f"Speech service device: {self._device}")
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if the Whisper model is loaded."""
        return self._pipeline is not None

    def load_model(self) -> None:
        """
        Load the Whisper model pipeline.

        Raises:
            SpeechServiceError: If model loading fails.
        """
        if self._pipeline is not None:
            logger.info("Whisper model already loaded")
            return

        try:
            import os
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{self.model_name.replace('/', '--')}"
            if cache_dir.exists():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                try:
                    import huggingface_hub.constants
                    huggingface_hub.constants.HF_HUB_OFFLINE = True
                except Exception:
                    pass

            from transformers import pipeline

            logger.info(
                f"Loading Whisper model: {self.model_name} "
                f"on {self.device}"
            )

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            )

            logger.info("Whisper model loaded successfully")

        except Exception as e:
            self._pipeline = None
            raise SpeechServiceError(
                f"Failed to load Whisper model '{self.model_name}': {e}"
            ) from e

    def unload_model(self) -> None:
        """Unload the Whisper model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("Whisper model unloaded")

    def transcribe(
        self,
        audio_path: str | Path,
        return_timestamps: bool = True,
        chunk_length_s: int = 30,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using Whisper.

        Args:
            audio_path: Path to the audio file.
            return_timestamps: Whether to include word/segment timestamps.
            chunk_length_s: Chunk length for long audio processing.

        Returns:
            TranscriptionResult with text and optional timestamps.

        Raises:
            SpeechServiceError: If transcription fails.
        """
        if self._pipeline is None:
            raise SpeechServiceError(
                "Whisper model not loaded. Call load_model() first."
            )

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise SpeechServiceError(
                f"Audio file not found: {audio_path}"
            )

        try:
            logger.info(f"Transcribing: {audio_path.name}")

            result = self._pipeline(
                str(audio_path),
                return_timestamps=return_timestamps,
                chunk_length_s=chunk_length_s,
            )

            # Extract full text
            full_text = result.get("text", "").strip()

            # Extract timestamped segments
            segments = []
            chunks = result.get("chunks", [])

            for chunk in chunks:
                chunk_text = chunk.get("text", "").strip()
                if not chunk_text:
                    continue
                ts = chunk.get("timestamp", (None, None))
                start = float(ts[0]) if ts[0] is not None else 0.0
                end = float(ts[1]) if (ts[1] is not None and ts[1] >= start) else start

                segments.append(
                    TimestampedSegment(
                        text=chunk_text,
                        start=start,
                        end=end,
                    )
                )

            # Fallback if no chunks were extracted but full text is available
            if not segments and full_text:
                segments.append(
                    TimestampedSegment(
                        text=full_text,
                        start=0.0,
                        end=0.0,
                    )
                )

            logger.info(
                f"Transcription complete: {len(full_text)} chars, "
                f"{len(segments)} segments"
            )

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                language="en",
            )

        except SpeechServiceError:
            raise
        except Exception as e:
            raise SpeechServiceError(
                f"Transcription failed for '{audio_path.name}': {e}"
            ) from e
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
