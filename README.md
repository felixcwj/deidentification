# Deidentification

A practical, Windows-first toolkit to **de-identify faces in images and MP4 videos** while keeping the original **hair and ears** intact. The anonymized face should **track movement** and **preserve expressions/emotions**.  
When a clip contains **2+ people**, the system **assigns a unique fake face to each person** and keeps it **consistent** throughout the video.

## Core Requirements

- Inputs: image (.jpg/.png) or video (.mp4).
- Multi-person: track individuals and keep a **stable de-identified face per person**.
- Preserve: **hair + ears**. Only the **facial skin/features** change.
- Expressions: retain **facial expression dynamics** as much as possible.
- Platform: Windows (PowerShell automation).
- Preferred engine: [hanweikung/face_anon_simple](https://github.com/hanweikung/face_anon_simple) (free/OSS).
- GPU: Prefer GPU acceleration; fall back to CPU when not available.

## Planned Pipeline (High-level)

1. **Frame IO**
   - Extract frames from MP4 (ffmpeg).
   - Reassemble frames + original audio after anonymization.

2. **Face Detection & Tracking**
   - Detect faces on frames; build tracks across cuts/camera changes.
   - Assign a **stable synthetic identity** per track.

3. **Face Swap / De-ID**
   - Swap **facial region only** using InsightFace models (hair/ears untouched).
   - Blend for natural results and expression retention.

4. **Consistency**
   - Maintain per-person identity across the entire clip (even after shot changes).

5. **Output**
   - Images: write anonymized image(s).
   - Videos: re-encode video; **mux original audio**.

## Roadmap

- v0.1: Windows PowerShell wrapper + minimal CLI; single-person basic flow.
- v0.2: Multi-person identity locking; robust shot changes.
- v0.3: GPU acceleration toggles (DirectML for models, NVENC/NVDEC for video IO).
- v0.4: Quality controls (masks, blend strength, CRF/bitrate presets).
- v0.5: Batch mode; logging; simple tests.

## Usage (to be implemented)

- **PowerShell wrapper**: one command to process either an image or an MP4.
- **Python CLI**: --image or --video plus --out, optional GPU flags.
- **Config**: env vars / flags for quality, speed, GPU/CPU preferences.

## Notes

- Target: free tooling; prefer OSS models and repos.
- Respect platform constraints (PowerShell 5 compatibility on Windows 10+).
- Avoid editing hair/ears by using appropriate face masks and swaps.

## License

TBD (MIT suggested).
