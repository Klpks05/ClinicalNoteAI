import pytest
import wave
import struct
from pathlib import Path

from backend.app.utils.audio_utils import (
    validate_audio_file,
    get_audio_duration,
    format_duration,
    format_file_size,
    AudioValidationError,
)

def test_format_duration():
    assert format_duration(30) == "0:30"
    assert format_duration(65) == "1:05"
    assert format_duration(3665) == "1:01:05"

def test_format_file_size():
    assert format_file_size(500) == "500 B"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(15 * 1024 * 1024) == "15.0 MB"

def test_validate_nonexistent_file():
    with pytest.raises(AudioValidationError, match="File not found"):
        validate_audio_file("nonexistent_file.wav")

def test_validate_unsupported_extension(tmp_path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not an audio file")
    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        validate_audio_file(bad_file)

def test_validate_empty_file(tmp_path):
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()
    with pytest.raises(AudioValidationError, match="empty"):
        validate_audio_file(empty_file)

def test_validate_real_primock_wav(sample_wav_path):
    if not sample_wav_path.exists():
        pytest.skip("PriMock57 sample WAV not found")
    info = validate_audio_file(sample_wav_path)
    assert info.filename == "day1_consultation01_doctor.wav"
    assert info.format == "wav"
    assert info.file_size_bytes > 0
    assert info.duration_seconds is not None
    assert info.duration_seconds > 0
    assert info.sample_rate is not None

def test_timestamped_segment_serialization():
    from backend.app.services.speech_service import TimestampedSegment
    seg = TimestampedSegment(text="Hello world", start=1.5, end=3.0, speaker="Doctor")
    d = seg.to_dict()
    assert d == {"text": "Hello world", "start": 1.5, "end": 3.0, "speaker": "Doctor"}
    # Verify model_dump compatibility alias
    assert seg.model_dump() == d

def test_ensure_audio_for_whisper(tmp_path):
    import soundfile as sf
    import numpy as np
    from backend.app.utils.audio_utils import ensure_audio_for_whisper

    # Create a 44.1kHz stereo WAV
    test_wav = tmp_path / "stereo_44k.wav"
    stereo_data = np.zeros((44100, 2), dtype=np.float32)
    sf.write(str(test_wav), stereo_data, 44100)

    whisper_path, is_temp = ensure_audio_for_whisper(test_wav)
    assert is_temp is True
    assert whisper_path.exists()

    info = sf.info(str(whisper_path))
    assert info.samplerate == 16000
    assert info.channels == 1

    # Cleanup
    if is_temp:
        whisper_path.unlink()
