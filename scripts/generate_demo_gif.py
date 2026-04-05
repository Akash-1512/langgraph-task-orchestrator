"""
scripts/generate_demo_gif.py

Automated demo GIF generator for langgraph-task-orchestrator.
- Starts FastAPI and Streamlit servers
- Uses Playwright to automate the browser
- Captures screenshots at each step
- Assembles into demo.gif for the README

Run with SYSTEM Python (not venv):
    python scripts/generate_demo_gif.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from PIL import Image

VENV_PYTHON = str(Path(".venv/Scripts/python.exe").resolve())
SCREENSHOTS_DIR = Path("scripts/screenshots")
OUTPUT_GIF = Path("docs/demo.gif")

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)


def start_servers():
    """Start FastAPI and Streamlit in background subprocesses."""
    print("🚀 Starting FastAPI server...")
    api = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "api.main:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("🚀 Starting Streamlit server...")
    ui = subprocess.Popen(
        [VENV_PYTHON, "-m", "streamlit", "run", "ui/app.py",
         "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("⏳ Waiting 8 seconds for servers to start...")
    time.sleep(8)
    return api, ui


def take_screenshots(page):
    """Automate the browser and capture screenshots at each step."""
    screenshots = []

    # Step 1 — Load the UI
    print("📸 Step 1: Loading UI...")
    page.goto("http://localhost:8501")
    page.wait_for_timeout(4000)
    path = str(SCREENSHOTS_DIR / "01_loaded.png")
    page.screenshot(path=path, full_page=False)
    screenshots.append(path)

    # Step 2 — Clear and type query
    print("📸 Step 2: Typing query...")
    try:
        textarea = page.locator("textarea").first
        textarea.click()
        textarea.fill("Analyze Apple's revenue performance from their latest SEC 10-K filing")
        page.wait_for_timeout(1000)
        path = str(SCREENSHOTS_DIR / "02_query_typed.png")
        page.screenshot(path=path, full_page=False)
        screenshots.append(path)
    except Exception as e:
        print(f"   ⚠️  Could not type query: {e}")

    # Step 3 — Click Run
    print("📸 Step 3: Clicking Run Agent Graph...")
    try:
        run_btn = page.get_by_text("Run Agent Graph").first
        run_btn.click()
        page.wait_for_timeout(2000)
        path = str(SCREENSHOTS_DIR / "03_running.png")
        page.screenshot(path=path, full_page=False)
        screenshots.append(path)
    except Exception as e:
        print(f"   ⚠️  Could not click run: {e}")

    # Step 4 — Wait for HITL (up to 90 seconds)
    print("⏳ Waiting for agent pipeline to complete (up to 90s)...")
    try:
        page.wait_for_selector("text=Approve & Finalize", timeout=90000)
        path = str(SCREENSHOTS_DIR / "04_hitl_ready.png")
        page.screenshot(path=path, full_page=False)
        screenshots.append(path)
        print("📸 Step 4: HITL checkpoint reached")
    except Exception as e:
        print(f"   ⚠️  HITL not reached in time: {e}")
        path = str(SCREENSHOTS_DIR / "04_timeout.png")
        page.screenshot(path=path, full_page=False)
        screenshots.append(path)

    # Step 5 — Approve
    print("📸 Step 5: Approving output...")
    try:
        approve_btn = page.get_by_text("Approve & Finalize").first
        approve_btn.click()
        page.wait_for_timeout(5000)
        path = str(SCREENSHOTS_DIR / "05_approved.png")
        page.screenshot(path=path, full_page=False)
        screenshots.append(path)
    except Exception as e:
        print(f"   ⚠️  Could not approve: {e}")

    return screenshots


def make_gif(screenshot_paths: list, output_path: Path):
    """Convert screenshots to animated GIF."""
    if not screenshot_paths:
        print("❌ No screenshots to convert")
        return

    images = []
    for path in screenshot_paths:
        if os.path.exists(path):
            img = Image.open(path)
            # Resize to 1280x720 for consistent GIF
            img = img.resize((1280, 720), Image.LANCZOS)
            images.append(img)

    if not images:
        print("❌ Could not load screenshots")
        return

    # Save as GIF — 2.5 seconds per frame
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=2500,
        loop=0,
        optimize=True,
    )
    print(f"✅ GIF saved to {output_path} ({len(images)} frames)")


def main():
    from playwright.sync_api import sync_playwright

    api_proc, ui_proc = start_servers()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 720})

            screenshots = take_screenshots(page)
            browser.close()

        make_gif(screenshots, OUTPUT_GIF)
        print(f"\n✅ Demo GIF generated: {OUTPUT_GIF}")
        print(f"   Add to README: ![Demo](docs/demo.gif)")

    finally:
        print("🛑 Stopping servers...")
        api_proc.terminate()
        ui_proc.terminate()


if __name__ == "__main__":
    main()