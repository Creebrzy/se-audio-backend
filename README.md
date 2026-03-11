# Sound Engine v4.7 — Audio Analysis Service

Ephemeral yt-dlp → FFmpeg → Librosa pipeline.  
Accepts a URL, extracts audio, runs music analysis, returns JSON, deletes audio.

---

## Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| API | FastAPI + Uvicorn | HTTP server |
| Extraction | yt-dlp | Pull audio stream from YouTube / URLs |
| Conversion | FFmpeg | Transcode to mono WAV 22050Hz |
| Analysis | Librosa + NumPy | BPM, key, energy, danceability, spectral |
| Deploy | Railway or Render | Container hosting |

**Target cost: $0–$10/month**

---

## API

### `GET /health`
Returns service status + confirms yt-dlp and ffmpeg are installed.

```json
{
  "status": "ok",
  "version": "4.7.0",
  "tools": { "yt-dlp": "ok", "ffmpeg": "ok" }
}
```

### `POST /analyze`

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=EXAMPLE",
  "include_waveform": false
}
```

**Response:**
```json
{
  "success": true,
  "track_id": "a3f9b2c1",
  "title": "Track Title",
  "artist": "Artist Name",
  "duration_seconds": 213.0,
  "bpm": 97.5,
  "key": "F#",
  "mode": "minor",
  "key_confidence": 0.82,
  "energy": 0.74,
  "danceability": 0.81,
  "loudness_db": -9.2,
  "spectral_centroid_mean": 2841.3,
  "tempo_stability": "stable",
  "genre_hints": ["Hip-Hop / Trap", "Pop / Dance-Pop"],
  "waveform": null
}
```

Set `include_waveform: true` to receive 200-point normalized amplitude array for UI display.

---

## Pipeline

```
User URL
  ↓
yt-dlp --dump-json        → title, artist, duration (no download)
  ↓
yt-dlp -f bestaudio -o -  → audio stream piped to stdout
  ↓
ffmpeg -i pipe:0 -ac 1 -ar 22050 -f wav
  ↓
librosa.load()            → numpy array in memory
  ↓
Analysis:
  • beat_track()          → BPM + tempo stability
  • chroma_cqt()          → key detection (Krumhansl-Schmuckler)
  • rms() + log10        → loudness dB
  • onset_strength()      → danceability heuristic
  • spectral_centroid()   → brightness / genre signal
  ↓
JSON returned
  ↓
WAV file deleted immediately (finally block — always runs)
```

---

## Local Development

```bash
# Install system deps (macOS)
brew install ffmpeg yt-dlp

# Install system deps (Ubuntu/Debian)
apt-get install ffmpeg
pip install yt-dlp

# Python setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run dev server
uvicorn main:app --reload --port 8000

# Test
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

---

## Deploy — Railway (Recommended, ~$5/mo)

1. Push this folder to a GitHub repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects the Dockerfile
4. Set env var: `ALLOWED_ORIGINS=https://soundengine3.pages.dev`
5. Copy the generated URL (e.g. `https://se-audio-api.up.railway.app`)
6. Paste it into the Sound Engine PWA as `SE_AUDIO_API_URL`

## Deploy — Render (~$7/mo)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → Connect repo
3. Render detects `render.yaml` automatically
4. Deploy — takes ~5 min first build (Librosa + FFmpeg are large)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `MAX_DURATION_SECONDS` | `600` | Max track length in seconds |
| `PORT` | `8000` | Set automatically by Railway/Render |

---

## Connecting to Sound Engine PWA

In the Sound Engine `_worker.js` or directly in `index.html`, set:

```javascript
const SE_AUDIO_API = 'https://your-service.up.railway.app';
```

The Mix Scan panel calls `SE_AUDIO_API + '/analyze'` with the user's URL.

---

## Future Expansion

The `/analyze` response schema is designed to be extended:

```json
// Future fields (not yet implemented):
{
  "stems": { ... },           // Demucs stem separation
  "chords": [ ... ],          // Chord progression timeline
  "sections": [ ... ],        // Verse/chorus/bridge markers
  "sample_candidates": [ ... ] // AI sample detection
}
```

Add new Librosa or model-based analysis to `analyze_wav()` in `main.py`  
without changing the API contract — new fields are additive.

---

## Cost Model

| Usage | Estimated Cost |
|-------|---------------|
| 0–100 analyses/month | Free tier (Railway/Render) |
| 100–1000/month | ~$5–7/mo (Starter plan) |
| Rate-limited free users (5/day) | Well within $10 budget |
| Producer tier (unlimited) | Monitor — add queue if needed |

---

## Notes

- Audio is **never stored permanently** — the `finally` block in `main.py` always runs cleanup
- Max track duration: 10 minutes (configurable via `MAX_DURATION_SECONDS`)
- Mono downmix + 22050Hz sample rate keeps processing fast (~5–15s per track)
- yt-dlp auto-updates its extractors — update periodically with `pip install -U yt-dlp`
