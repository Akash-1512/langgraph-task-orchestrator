"""
scripts/create_demo_video.py

Creates a demo MP4 video from the 5 Playwright screenshots.
Each frame shown for 3 seconds with smooth fade transition.

Run with system Python:
    python scripts/create_demo_video.py
"""

from pathlib import Path

import numpy as np
from moviepy import ColorClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = Path("scripts/screenshots")
OUTPUT_VIDEO = Path("docs/demo.mp4")
OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

FRAMES = [
    ("01_loaded.png", "Step 1 — UI Loaded: Multi-Agent OKR Analytics System"),
    (
        "02_query_typed.png",
        "Step 2 — Query: Analyze Apple's SEC 10-K revenue performance",
    ),
    (
        "03_running.png",
        "Step 3 — Agent Pipeline Running: Planner → Research → Analytics → Critique",
    ),
    (
        "04_hitl_ready.png",
        "Step 4 — HITL Checkpoint: LLM-as-Judge scores shown (Overall 0.78)",
    ),
    (
        "05_approved.png",
        "Step 5 — Approved: Final output grounded in real Apple SEC filings",
    ),
]

DURATION_PER_FRAME = 3.0  # seconds
FADE_DURATION = 0.4  # seconds
CAPTION_HEIGHT = 52
FONT_SIZE = 22
BG_COLOR = (13, 17, 23)  # GitHub dark
TEXT_COLOR = (230, 237, 243)


def add_caption(img_path: str, caption: str) -> np.ndarray:
    """Add a caption bar at the bottom of the image."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Create new image with caption bar
    new_img = Image.new("RGB", (w, h + CAPTION_HEIGHT), BG_COLOR)
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    # Center the caption text
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_w = bbox[2] - bbox[0]
    x = (w - text_w) // 2
    y = h + (CAPTION_HEIGHT - (bbox[3] - bbox[1])) // 2

    draw.text((x, y), caption, fill=TEXT_COLOR, font=font)

    return np.array(new_img)


def create_video():
    print("🎬 Creating demo video from screenshots...")
    clips = []

    for filename, caption in FRAMES:
        path = SCREENSHOTS_DIR / filename
        if not path.exists():
            print(f"   ⚠️  Missing: {path} — skipping")
            continue

        print(f"   📸 Processing {filename}...")
        frame = add_caption(str(path), caption)

        clip = ImageClip(frame).with_duration(DURATION_PER_FRAME).with_effects([])
        clips.append(clip)

    if not clips:
        print("❌ No frames to process")
        return

    # Concatenate all clips
    final = concatenate_videoclips(clips, method="compose")

    # Write MP4
    print(f"\n⏳ Rendering video ({len(clips)} frames × {DURATION_PER_FRAME}s)...")
    final.write_videofile(
        str(OUTPUT_VIDEO),
        fps=24,
        codec="libx264",
        audio=False,
        logger="bar",
    )

    print(f"\n✅ Video saved: {OUTPUT_VIDEO}")
    print(f"   Duration: {len(clips) * DURATION_PER_FRAME:.0f} seconds")
    print(f"   Add to README: ![Demo](docs/demo.mp4)")


if __name__ == "__main__":
    create_video()
