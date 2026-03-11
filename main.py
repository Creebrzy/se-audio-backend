"""
Sound Engine v4.7 — Audio Analysis Service
Ephemeral yt-dlp → FFmpeg → Librosa pipeline
Deploy to: Railway / Render / Fly.io
Cost target: $0–$10/month
"""

import os
import uuid
import json
import asyncio
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("se-audio")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sound Engine Audio Analysis API",
    version="4.7.0",
    description="Ephemeral audio extraction and music analysis service"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ─────────────────────────────────────────────────────────────────────
TEMP_DIR     = Path(tempfile.gettempdir()) / "se_audio"
MAX_DURATION = 600   # 10 min cap — refuse longer tracks
SAMPLE_RATE  = 22050 # librosa default

TEMP_DIR.mkdir(exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url: str
    include_waveform: bool = False   # waveform adds ~50ms, skip for basic analysis

class AnalysisResult(BaseModel):
    success: bool
    track_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    duration_seconds: Optional[float] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    key_confidence: Optional[float] = None
    mode: Optional[str] = None       # major / minor
    energy: Optional[float] = None   # 0–1
    danceability: Optional[float] = None  # 0–1
    loudness_db: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None
    tempo_stability: Optional[str] = None  # stable / variable
    genre_hints: Optional[list] = None
    waveform: Optional[list] = None  # downsampled for UI display
    error: Optional[str] = None

# ── Key detection helpers ──────────────────────────────────────────────────────
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
MINOR_PROFILE = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

def detect_key(y, sr):
    """Chroma-based key detection using Krumhansl-Schmuckler profiles."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_key, best_mode, best_corr = 0, 'major', -np.inf

    for root in range(12):
        # Rotate profiles to match root
        maj_rotated = np.roll(MAJOR_PROFILE, root)
        min_rotated = np.roll(MINOR_PROFILE, root)

        corr_maj = np.corrcoef(chroma_mean, maj_rotated)[0, 1]
        corr_min = np.corrcoef(chroma_mean, min_rotated)[0, 1]

        if corr_maj > best_corr:
            best_corr, best_key, best_mode = corr_maj, root, 'major'
        if corr_min > best_corr:
            best_corr, best_key, best_mode = corr_min, root, 'minor'

    confidence = min(1.0, max(0.0, (best_corr + 1) / 2))
    return PITCH_CLASSES[best_key], best_mode, round(confidence, 3)

def estimate_energy(y):
    """RMS-based energy normalized to 0–1."""
    rms = librosa.feature.rms(y=y)[0]
    raw = float(np.mean(rms))
    # Normalize: typical music RMS ~0.05–0.3
    return round(min(1.0, raw / 0.25), 3)

def estimate_danceability(y, sr, tempo):
    """
    Heuristic danceability: combines tempo regularity + onset strength consistency.
    High = steady beat in 90–130 BPM range.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Autocorrelation of onset envelope measures beat regularity
    ac = librosa.autocorrelate(onset_env, max_size=sr // 512)
    ac_norm = librosa.util.normalize(ac)

    # Score: peak AC in the beat-period window
    beat_period = max(1, int(round(60.0 / tempo * sr / 512)))
    window = ac_norm[max(0, beat_period - 3): beat_period + 4]
    regularity = float(np.max(window)) if len(window) > 0 else 0.5

    # Tempo sweet spot bonus: 90–130 BPM
    tempo_score = 1.0 if 90 <= tempo <= 130 else (0.8 if 70 <= tempo <= 150 else 0.5)

    danceability = round((regularity * 0.7 + tempo_score * 0.3), 3)
    return min(1.0, max(0.0, danceability))

def genre_hints_from_features(bpm, energy, spectral_centroid, mode):
    """
    Rule-based genre hints — not ML classification, just useful signals.
    Returns up to 3 candidate genre families.
    """
    hints = []
    if bpm < 80:
        hints.append("Slow Jam / R&B Ballad")
    if 80 <= bpm < 100 and energy < 0.45:
        hints.append("Neo-Soul / Lo-Fi")
    if 80 <= bpm < 110 and energy > 0.55:
        hints.append("Hip-Hop / Trap")
    if 110 <= bpm < 130 and energy > 0.6:
        hints.append("Pop / Dance-Pop")
    if 125 <= bpm <= 140 and spectral_centroid > 3000:
        hints.append("House / Electronic")
    if bpm > 140:
        hints.append("Drum & Bass / Jungle")
    if spectral_centroid < 1800 and mode == 'minor' and energy < 0.5:
        hints.append("Dark Ambient / Atmospheric")
    if not hints:
        hints.append("Crossover / Undetermined")
    return hints[:3]

# ── Extraction ─────────────────────────────────────────────────────────────────
async def extract_audio(url: str, track_id: str) -> tuple[Path, dict]:
    """
    Run yt-dlp to get best audio stream, convert to WAV via FFmpeg.
    Returns (wav_path, metadata_dict).
    """
    wav_path = TEMP_DIR / f"{track_id}.wav"
    meta: dict = {}

    # Step 1: Get metadata (title, artist, duration) without downloading
    meta_cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        url
    ]
    try:
        meta_proc = await asyncio.create_subprocess_exec(
            *meta_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        meta_stdout, _ = await asyncio.wait_for(meta_proc.communicate(), timeout=30)
        if meta_stdout:
            info = json.loads(meta_stdout.decode())
            meta["title"]    = info.get("title", "Unknown")
            meta["artist"]   = info.get("artist") or info.get("uploader", "Unknown")
            meta["duration"] = info.get("duration", 0)

            # Enforce max duration
            if meta["duration"] and meta["duration"] > MAX_DURATION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Track too long ({meta['duration']}s). Max {MAX_DURATION}s."
                )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Metadata fetch timed out")

    # Step 2: Extract + convert to WAV in one piped command
    # yt-dlp writes best audio to stdout, FFmpeg reads from stdin
    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--no-playlist",
        "-o", "-",          # output to stdout
        "--quiet",
        url
    ]
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",     # read from stdin
        "-ac", "1",         # mono (faster analysis)
        "-ar", str(SAMPLE_RATE),
        "-f", "wav",
        "-y",
        str(wav_path)
    ]

    try:
        ytdlp_proc = await asyncio.create_subprocess_exec(
            *ytdlp_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        ffmpeg_proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin=ytdlp_proc.stdout,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )
        _, ffmpeg_err = await asyncio.wait_for(
            ffmpeg_proc.communicate(), timeout=120
        )
        await ytdlp_proc.wait()

        if ffmpeg_proc.returncode != 0:
            err_msg = ffmpeg_err.decode()[-300:] if ffmpeg_err else "FFmpeg failed"
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {err_msg}")

    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Audio extraction timed out (120s limit)")

    if not wav_path.exists() or wav_path.stat().st_size < 1000:
        raise HTTPException(status_code=500, detail="Extracted audio file is empty or missing")

    return wav_path, meta

# ── Analysis ───────────────────────────────────────────────────────────────────
def analyze_wav(wav_path: Path, include_waveform: bool = False) -> dict:
    """Run full Librosa analysis on WAV file. Returns feature dict."""

    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

    # ── BPM / Tempo ──
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(tempo), 1)

    # Tempo stability — check variance in inter-beat intervals
    if len(beat_frames) > 2:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        ibi = np.diff(beat_times)
        cv = np.std(ibi) / np.mean(ibi) if np.mean(ibi) > 0 else 1.0
        tempo_stability = "stable" if cv < 0.15 else "variable"
    else:
        tempo_stability = "unknown"

    # ── Key ──
    key, mode, key_confidence = detect_key(y, sr)

    # ── Energy ──
    energy = estimate_energy(y)

    # ── Danceability ──
    danceability = estimate_danceability(y, sr, bpm)

    # ── Loudness (LUFS approximation via RMS) ──
    rms = float(np.mean(librosa.feature.rms(y=y)))
    loudness_db = round(20 * np.log10(rms + 1e-9), 1)

    # ── Spectral Centroid ──
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = round(float(np.mean(sc)), 1)

    # ── Genre hints ──
    genre_hints = genre_hints_from_features(bpm, energy, spectral_centroid_mean, mode)

    result = {
        "bpm": bpm,
        "key": key,
        "mode": mode,
        "key_confidence": key_confidence,
        "energy": energy,
        "danceability": danceability,
        "loudness_db": loudness_db,
        "spectral_centroid_mean": spectral_centroid_mean,
        "tempo_stability": tempo_stability,
        "genre_hints": genre_hints,
    }

    # ── Optional waveform (downsampled to 200 points for UI) ──
    if include_waveform:
        waveform_full = np.abs(y)
        n = 200
        step = max(1, len(waveform_full) // n)
        waveform = [round(float(v), 4) for v in waveform_full[::step][:n]]
        result["waveform"] = waveform

    return result

# ── Cleanup ────────────────────────────────────────────────────────────────────
def cleanup(path: Path):
    """Delete temp audio file immediately after analysis."""
    try:
        if path.exists():
            path.unlink()
            log.info(f"Deleted temp file: {path.name}")
    except Exception as e:
        log.warning(f"Cleanup failed for {path}: {e}")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — also verifies yt-dlp and ffmpeg are available."""
    checks = {}
    for tool in ["yt-dlp", "ffmpeg"]:
        try:
            r = subprocess.run([tool, "--version"],
                               capture_output=True, timeout=5)
            checks[tool] = "ok" if r.returncode == 0 else "error"
        except FileNotFoundError:
            checks[tool] = "missing"
    return {"status": "ok", "version": "4.7.0", "tools": checks}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(req: AnalyzeRequest):
    """
    Main endpoint. Accepts a URL, extracts audio, runs analysis, returns JSON.
    Audio is deleted immediately after analysis.
    """
    track_id = str(uuid.uuid4())[:8]
    wav_path = TEMP_DIR / f"{track_id}.wav"
    log.info(f"[{track_id}] Starting analysis for: {req.url[:80]}")

    try:
        # 1. Extract audio
        wav_path, meta = await extract_audio(req.url, track_id)
        log.info(f"[{track_id}] Extracted: {wav_path.stat().st_size // 1024}KB")

        # 2. Analyze
        features = analyze_wav(wav_path, include_waveform=req.include_waveform)
        log.info(f"[{track_id}] Analysis complete: {features['bpm']} BPM, {features['key']} {features['mode']}")

        return AnalysisResult(
            success=True,
            track_id=track_id,
            title=meta.get("title"),
            artist=meta.get("artist"),
            duration_seconds=meta.get("duration"),
            **features
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[{track_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 3. Always delete audio — even on error
        cleanup(wav_path)
