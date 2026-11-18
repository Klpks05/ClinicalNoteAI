"""
Audio utility functions for file validation, format detection,
and metadata extraction.
"""

import os
import wave
import struct
from pathlib import Path
from dataclasses import dataclass


# Supported audio extensions
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

# Maximum audio file size (100 MB)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

# Maximum audio duration (60 minutes)
MAX_DURATION_SECONDS = 3600


@dataclass
class AudioInfo:
    """Metadata about an audio file."""

    file_path: str
    filename: str
    file_size_bytes: int
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    format: str


class AudioValidationError(Exception):
    """Raised when audio validation fails."""
    pass


def validate_audio_file(file_path: str | Path) -> AudioInfo:
    """
    Validate an audio file and return its metadata.

    Checks:
    - File exists
    - Extension is supported
    - File size is within limits
    - WAV files: validates header and duration

    Args:
        file_path: Path to the audio file.

    Returns:
        AudioInfo with file metadata.

    Raises:
        AudioValidationError: If the file is invalid.
    """
    path = Path(file_path)

    # Check file exists
    if not path.exists():
        raise AudioValidationError(f"File not found: {path}")

    if not path.is_file():
        raise AudioValidationError(f"Not a file: {path}")

    # Check extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported audio format: '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Check file size
    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioValidationError("File is empty (0 bytes)")

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise AudioValidationError(
            f"File too large: {size_mb:.1f} MB "
            f"(maximum: {max_mb:.0f} MB)"
        )

    # Extract metadata using robust fallback across wave, soundfile, and torchaudio
    duration, sample_rate, channels = _get_audio_metadata(path, ext)

    if duration is None or duration <= 0:
        raise AudioValidationError("Audio file duration is zero or invalid")

    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio too long: {duration:.0f}s "
            f"(maximum: {MAX_DURATION_SECONDS}s / "
            f"{MAX_DURATION_SECONDS // 60} minutes)"
        )

    if sample_rate is None or sample_rate <= 0:
        raise AudioValidationError("Invalid audio sample rate")

    return AudioInfo(
        file_path=str(path),
        filename=path.name,
        file_size_bytes=file_size,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        format=ext.lstrip("."),
    )


def _get_audio_metadata(path: Path, ext: str) -> tuple[float, int, int]:
    """
    Read audio metadata with fallback across wave, soundfile, and torchaudio/librosa.
    Returns (duration_seconds, sample_rate, num_channels).
    """
    last_err = None

    # For WAV files, first try native wave module
    if ext == ".wav":
        try:
            return _get_wav_info(str(path))
        except Exception as e:
            last_err = e

    # Try soundfile (supports WAV, FLAC, OGG, MP3, etc.)
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate), int(info.channels)
    except Exception as e:
        last_err = e

    # Try torchaudio (supports m4a, webm, mp3, wav, etc.)
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        sr = int(info.sample_rate)
        dur = float(info.num_frames) / float(sr) if sr > 0 else 0.0
        return dur, sr, int(info.num_channels)
    except Exception as e:
        last_err = e

    # Try librosa as last resort
    try:
        import librosa
        dur = float(librosa.get_duration(path=str(path)))
        sr = int(librosa.get_samplerate(str(path)))
        return dur, sr, 1
    except Exception as e:
        last_err = e

    raise AudioValidationError(
        f"Invalid or corrupted audio file '{path.name}': {last_err}"
    )


def ensure_audio_for_whisper(input_path: str | Path) -> tuple[Path, bool]:
    """
    Ensure audio is 16kHz mono WAV suitable for Whisper and acoustic diarization.

    If already 16kHz mono WAV, returns (input_path, False).
    Otherwise, resamples and downmixes to 16kHz mono 16-bit PCM WAV.
    Returns:
        (path_to_use, is_temporary)
    """
    path = Path(input_path)
    try:
        duration, sr, ch = _get_audio_metadata(path, path.suffix.lower())
        if path.suffix.lower() == ".wav" and sr == 16000 and ch == 1:
            return path, False
    except Exception:
        pass

    import soundfile as sf
    import librosa

    out_path = path.with_suffix(".whisper_16k.wav")
    try:
        y, _ = librosa.load(str(path), sr=16000, mono=True)
        sf.write(str(out_path), y, 16000, subtype="PCM_16", format="WAV")
        return out_path, True
    except Exception as e:
        raise AudioValidationError(
            f"Failed to convert audio '{path.name}' to 16kHz mono WAV: {e}"
        ) from e


def get_audio_duration(file_path: str | Path) -> float | None:
    """
    Get the duration of an audio file in seconds.
    """
    path = Path(file_path)
    try:
        dur, _, _ = _get_audio_metadata(path, path.suffix.lower())
        return dur
    except Exception:
        return None


def _get_wav_info(file_path: str) -> tuple[float, int, int]:
    """
    Read WAV file header and return (duration, sample_rate, channels).
    """
    with wave.open(file_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        channels = wf.getnchannels()

        if sample_rate == 0:
            raise ValueError("WAV file has sample rate of 0")

        duration = n_frames / sample_rate

    return duration, sample_rate, channels


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "5:30" or "1:02:15".
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to a human-readable string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string like "14.6 MB".
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
