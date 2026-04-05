# Demo Scripts

These scripts generate the demo video and GIF for the README.
Run with **system Python** (not venv) — requires moviepy, edge-tts, playwright.

## Setup
pip install moviepy edge-tts playwright pillow
python -m playwright install chromium

## Generate demo GIF
python scripts/generate_demo_gif.py
# Output: docs/demo.gif

## Generate narrated MP4
python scripts/create_narrated_video.py
# Output: docs/demo_narrated.mp4
# Requires: uvicorn + streamlit servers running (start them first)
