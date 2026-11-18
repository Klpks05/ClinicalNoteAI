#!/usr/bin/env python3
"""
Benchmark Whisper Speech Recognition on Apple Silicon M4.

Measures:
- Audio duration vs. transcription time
- Real-Time Factor (RTF = transcription_time / audio_duration)
- Model loading time (cold load)
- Device comparison: MPS vs CPU (on 60s sample)
- Batch size comparison: batch_size=1 vs 4 vs 8 (on full 7.6m audio)
- Model comparison: openai/whisper-small.en vs openai/whisper-base.en
- Memory footprint (RAM RSS & MPS allocated memory)
- Word Error Rate (WER) against PriMock57 TextGrid reference
- Timestamp preservation
"""

import os
import re
import time
import json
from pathlib import Path
import psutil
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    import huggingface_hub.constants
    huggingface_hub.constants.HF_HUB_OFFLINE = True
except Exception:
    pass

from transformers import pipeline
import soundfile as sf

AUDIO_FILE = Path("data/primock57/audio/day1_consultation01_doctor.wav")
TEXTGRID_FILE = Path("data/primock57/transcripts/day1_consultation01_doctor.TextGrid")
OUTPUT_BENCHMARK = Path("outputs/whisper_benchmark_results.json")

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def parse_textgrid_reference(tg_path: Path) -> str:
    """Extract reference text from PriMock57 TextGrid, removing tags."""
    lines = tg_path.read_text(encoding="utf-8").splitlines()
    texts = []
    for line in lines:
        line = line.strip()
        if line.startswith('text = '):
            val = line[7:].strip('"')
            # Clean up <UNIN/>, <UNSURE> tags
            val = re.sub(r"<[^>]+>", " ", val)
            val = re.sub(r"\s+", " ", val).strip()
            if val:
                texts.append(val)
    return " ".join(texts)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using simple Levenshtein distance on words."""
    ref_words = re.sub(r"[^\w\s]", "", reference.lower()).split()
    hyp_words = re.sub(r"[^\w\s]", "", hypothesis.lower()).split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r_len, h_len = len(ref_words), len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[r_len][h_len] / float(r_len)

def main():
    print("=" * 75)
    print("WHISPER PERFORMANCE PROFILING ON APPLE SILICON M4")
    print("=" * 75)

    info = sf.info(str(AUDIO_FILE))
    audio_duration = info.duration
    print(f"Audio file: {AUDIO_FILE.name}")
    print(f"Sample Rate: {info.samplerate} Hz, Channels: {info.channels}, Duration: {audio_duration:.2f} s ({audio_duration/60:.2f} min)")
    
    ref_text = parse_textgrid_reference(TEXTGRID_FILE)
    print(f"Reference transcript extracted: {len(ref_text.split())} words")

    results = {
        "audio_file": AUDIO_FILE.name,
        "audio_duration_seconds": round(audio_duration, 2),
        "mps_available": torch.backends.mps.is_available(),
        "experiments": []
    }

    # 1. Device Speed Comparison on 60-second snippet (MPS vs CPU)
    print("\n--- 1. HARDWARE ACCELERATION: MPS vs CPU (60s snippet) ---")
    data, sr = sf.read(str(AUDIO_FILE))
    snippet_60s = data[:sr * 60]
    snippet_path = Path("outputs/temp_snippet_60s.wav")
    sf.write(str(snippet_path), snippet_60s, sr)

    for device in (["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]):
        print(f"\nTesting device={device} on 60s audio with whisper-small.en...")
        t0 = time.time()
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small.en",
            device=device,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        )
        load_time = time.time() - t0
        
        t0 = time.time()
        res = pipe(str(snippet_path), return_timestamps=True, chunk_length_s=30, batch_size=1)
        transcribe_time = time.time() - t0
        rtf = transcribe_time / 60.0

        mem_mb = get_process_memory_mb()
        print(f"  Device: {device} | Load Time: {load_time:.2f}s | Transcribe: {transcribe_time:.2f}s | RTF: {rtf:.3f} | RAM: {mem_mb:.1f} MB")
        
        results["experiments"].append({
            "name": f"device_comparison_{device}",
            "device": device,
            "model": "openai/whisper-small.en",
            "audio_duration": 60.0,
            "batch_size": 1,
            "load_time_seconds": round(load_time, 2),
            "transcribe_time_seconds": round(transcribe_time, 2),
            "rtf": round(rtf, 4),
            "ram_mb": round(mem_mb, 1),
        })

        del pipe
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if snippet_path.exists():
        snippet_path.unlink()

    # 2. Batching Optimization on Full Audio (458s ~ 7.6m) on MPS
    print("\n--- 2. BATCHING & CHUNKING OPTIMIZATION (Full 7.6 min audio on MPS) ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    t0 = time.time()
    pipe_small = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",
        device=device,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    )
    small_load_time = time.time() - t0
    print(f"Cold model load time (whisper-small.en, {device}): {small_load_time:.2f}s")

    for batch_size in [1, 4, 8]:
        print(f"\nRunning whisper-small.en with batch_size={batch_size}...")
        t0 = time.time()
        res = pipe_small(
            str(AUDIO_FILE),
            return_timestamps=True,
            chunk_length_s=30,
            batch_size=batch_size,
        )
        transcribe_time = time.time() - t0
        rtf = transcribe_time / audio_duration
        chunks = res.get("chunks", [])
        num_segments = len(chunks)
        hyp_text = res.get("text", "")
        wer = calculate_wer(ref_text, hyp_text)
        mem_mb = get_process_memory_mb()

        has_timestamps = len(chunks) > 0 and chunks[0].get("timestamp") is not None
        first_ts = chunks[0]["timestamp"] if has_timestamps else None
        last_ts = chunks[-1]["timestamp"] if has_timestamps else None

        print(f"  batch_size={batch_size}: Transcribe Time: {transcribe_time:.2f}s | RTF: {rtf:.3f} | WER: {wer:.4f} ({wer*100:.1f}%) | Segments: {num_segments} | RAM: {mem_mb:.1f} MB")
        print(f"  Timestamps intact: {has_timestamps} (First: {first_ts}, Last: {last_ts})")

        results["experiments"].append({
            "name": f"batching_bs_{batch_size}",
            "device": device,
            "model": "openai/whisper-small.en",
            "audio_duration": round(audio_duration, 2),
            "batch_size": batch_size,
            "transcribe_time_seconds": round(transcribe_time, 2),
            "rtf": round(rtf, 4),
            "wer": round(wer, 4),
            "num_segments": num_segments,
            "timestamps_intact": has_timestamps,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "ram_mb": round(mem_mb, 1),
        })

    del pipe_small
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 3. Model Size Comparison: whisper-base.en vs whisper-small.en (both batch_size=8)
    print("\n--- 3. MODEL COMPARISON: whisper-base.en vs whisper-small.en ---")
    t0 = time.time()
    pipe_base = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base.en",
        device=device,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    )
    base_load_time = time.time() - t0
    print(f"Cold model load time (whisper-base.en, {device}): {base_load_time:.2f}s")

    t0 = time.time()
    res_base = pipe_base(
        str(AUDIO_FILE),
        return_timestamps=True,
        chunk_length_s=30,
        batch_size=8,
    )
    base_transcribe_time = time.time() - t0
    base_rtf = base_transcribe_time / audio_duration
    base_hyp_text = res_base.get("text", "")
    base_wer = calculate_wer(ref_text, base_hyp_text)
    base_mem_mb = get_process_memory_mb()

    print(f"  whisper-base.en: Transcribe Time: {base_transcribe_time:.2f}s | RTF: {base_rtf:.3f} | WER: {base_wer:.4f} ({base_wer*100:.1f}%) | RAM: {base_mem_mb:.1f} MB")

    results["experiments"].append({
        "name": "model_comparison_whisper_base",
        "device": device,
        "model": "openai/whisper-base.en",
        "audio_duration": round(audio_duration, 2),
        "batch_size": 8,
        "load_time_seconds": round(base_load_time, 2),
        "transcribe_time_seconds": round(base_transcribe_time, 2),
        "rtf": round(base_rtf, 4),
        "wer": round(base_wer, 4),
        "ram_mb": round(base_mem_mb, 1),
    })

    OUTPUT_BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BENCHMARK, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"PROFILING COMPLETE! Results saved to {OUTPUT_BENCHMARK}")
    print("=" * 75)

if __name__ == "__main__":
    main()
