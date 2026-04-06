"""
scripts/create_narrated_video.py

Creates a fully narrated demo MP4 with:
- Premium Mermaid-style architecture flow diagram with animated highlights
- AI narration via Microsoft Edge TTS (en-US-AndrewNeural)
- Word-level subtitle sync (real boundaries from edge-tts)
- On-screen floating annotations per highlighted node
- Yellow glow highlights on active nodes
- Tech Stack slide with card grid
- 5 live demo frames

Run with system Python:
    python scripts/create_narrated_video.py

Audit fixes applied (v2):
- Flaw 2: AudioFileClip.close() called after reading duration
- Flaw 3: TTS writes to .tmp file; renamed only on success
- Flaw 4: draw_glow single-pass composite (no per-layer img mutation)
- Flaw 5: Settings uses __init__ (see config.py)
- Flaw 6: scripts/__init__.py added
"""

import asyncio
import json
import math
from pathlib import Path

import edge_tts
import numpy as np
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────────
SCREENSHOTS_DIR = Path("scripts/screenshots")
AUDIO_DIR = Path("scripts/audio")
OUTPUT_VIDEO = Path("docs/demo_narrated.mp4")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

VOICE = "en-US-AndrewNeural"
FPS = 24
CAPTION_HEIGHT = 64
W, H = 1280, 720

# ── Premium palette ────────────────────────────────────────────────────────────
BG = (10, 14, 20)
NODE_TOP = (24, 34, 50)
NODE_BOT = (16, 22, 34)
NODE_BORDER = (40, 52, 68)
ARROW_CLR = (56, 139, 253)
TXT_PRI = (230, 237, 243)
TXT_SEC = (120, 140, 160)
ACCENT = (255, 213, 0)
ACCENT_DIM = (255, 213, 0, 38)
SUCCESS = (48, 175, 90)
DANGER = (248, 81, 73)
CAPTION_BG = (14, 18, 26)

HIGHLIGHT_COLOR = ACCENT
HIGHLIGHT_FILL = ACCENT_DIM

# ── Node layout constants ──────────────────────────────────────────────────────
NW, NH = 180, 80
N = {
    "planner": (175, 265),
    "research": (415, 265),
    "analytics": (655, 265),
    "critique": (895, 265),
    "hitl": (655, 455),
    "end": (655, 590),
    "supervisor": (1100, 360),
}

NODE_BOXES = {
    "Planner Agent": (
        N["planner"][0] - NW // 2,
        N["planner"][1] - NH // 2,
        N["planner"][0] + NW // 2,
        N["planner"][1] + NH // 2,
    ),
    "Research Agent": (
        N["research"][0] - NW // 2,
        N["research"][1] - NH // 2,
        N["research"][0] + NW // 2,
        N["research"][1] + NH // 2,
    ),
    "Analytics Agent": (
        N["analytics"][0] - NW // 2,
        N["analytics"][1] - NH // 2,
        N["analytics"][0] + NW // 2,
        N["analytics"][1] + NH // 2,
    ),
    "Critique Agent": (
        N["critique"][0] - NW // 2,
        N["critique"][1] - NH // 2,
        N["critique"][0] + NW // 2,
        N["critique"][1] + NH // 2,
    ),
    "HITL Checkpoint": (
        N["hitl"][0] - NW // 2,
        N["hitl"][1] - NH // 2,
        N["hitl"][0] + NW // 2,
        N["hitl"][1] + NH // 2,
    ),
    "Supervisor Agent": (
        N["supervisor"][0] - 130,
        N["supervisor"][1] - 38,
        N["supervisor"][0] + 130,
        N["supervisor"][1] + 38,
    ),
}

NODE_ANNOTATIONS = {
    "Planner Agent": [
        "Reads: query",
        "Writes: plan (5 sub-tasks)",
        "LLM: llama-3.3-70b",
    ],
    "Research Agent": [
        "Reads: plan",
        "Retrieves: top-4 chunks via RAG",
        "Source: 2,284 SEC EDGAR docs",
    ],
    "Analytics Agent": [
        "Reads: context + plan",
        "Writes: structured report",
        "Rule: grounded — no hallucination",
    ],
    "Critique Agent": [
        "LLM-as-Judge scoring",
        "Faithfulness · Coherence · Task",
        "Gate: overall >= 0.75",
    ],
    "HITL Checkpoint": [
        "interrupt() pauses graph",
        "State serialised to SQLite",
        "Resumes: approve / revise",
    ],
    "Supervisor Agent": [
        "Meta-router node",
        "Evaluates state completeness",
        "Routes to next best agent",
    ],
}


# ── Font helper ────────────────────────────────────────────────────────────────
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Drawing primitives ─────────────────────────────────────────────────────────


def draw_dot_grid(draw: ImageDraw.Draw, w: int, h: int, spacing: int = 32) -> None:
    for x in range(0, w, spacing):
        for y in range(0, h, spacing):
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(22, 30, 44))


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_rounded_gradient(
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    top_c: tuple,
    bot_c: tuple,
    radius: int = 10,
    border_c: tuple = None,
    border_w: int = 1,
) -> None:
    """Rounded rectangle with vertical gradient fill."""
    w, h = x2 - x1, y2 - y1
    tile = Image.new("RGB", (w, h), top_c)
    td = ImageDraw.Draw(tile)
    for row in range(h):
        td.line(
            [(0, row), (w, row)], fill=lerp_color(top_c, bot_c, row / max(h - 1, 1))
        )
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, w - 1, h - 1], radius=radius, fill=255
    )
    tile.putalpha(mask)
    img.paste(tile, (x1, y1), tile)
    if border_c:
        d = ImageDraw.Draw(img)
        for i in range(border_w):
            d.rounded_rectangle(
                [x1 + i, y1 + i, x2 - i, y2 - i], radius=radius, outline=border_c
            )


def draw_glow(
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple = (255, 213, 0),
    layers: int = 5,
) -> None:
    """
    FIX (Flaw 4): Single-pass composite — all glow layers drawn onto one
    RGBA overlay, then composited once. Eliminates per-layer img mutation loop.
    """
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(layers, 0, -1):
        pad = i * 4
        alpha = int(50 * (1 - i / (layers + 1)))
        gd.rounded_rectangle(
            [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
            radius=14 + pad,
            fill=color + (alpha,),
        )
    base = img.convert("RGBA")
    img.paste(Image.alpha_composite(base, glow).convert("RGB"))


def arrowhead(
    draw: ImageDraw.Draw,
    tx: int,
    ty: int,
    angle_deg: float,
    size: int = 9,
    color: tuple = ARROW_CLR,
) -> None:
    a = math.radians(angle_deg)
    sp = math.radians(28)
    pts = [
        (tx, ty),
        (tx - size * math.cos(a - sp), ty - size * math.sin(a - sp)),
        (tx - size * math.cos(a + sp), ty - size * math.sin(a + sp)),
    ]
    draw.polygon(pts, fill=color)


def draw_h_arrow(
    draw: ImageDraw.Draw,
    x1: int,
    y: int,
    x2: int,
    color: tuple = ARROW_CLR,
    label: str = None,
) -> None:
    draw.line([(x1, y), (x2, y)], fill=color, width=2)
    arrowhead(draw, x2, y, 0 if x2 > x1 else 180, color=color)
    if label:
        f = get_font(12)
        lx = (x1 + x2) // 2
        ly = y - 16
        bb = draw.textbbox((0, 0), label, font=f)
        tw = bb[2] - bb[0]
        draw.rectangle(
            [lx - tw // 2 - 3, ly - 2, lx + tw // 2 + 3, ly + bb[3] - bb[1] + 2],
            fill=(18, 24, 36),
        )
        draw.text((lx - tw // 2, ly), label, fill=TXT_SEC, font=f)


def draw_v_arrow(
    draw: ImageDraw.Draw,
    x: int,
    y1: int,
    y2: int,
    color: tuple = ARROW_CLR,
    label: str = None,
) -> None:
    draw.line([(x, y1), (x, y2)], fill=color, width=2)
    arrowhead(draw, x, y2, 90 if y2 > y1 else 270, color=color)
    if label:
        f = get_font(12)
        lx = x + 8
        ly = (y1 + y2) // 2
        bb = draw.textbbox((0, 0), label, font=f)
        draw.rectangle(
            [lx - 2, ly - 2, lx + bb[2] - bb[0] + 4, ly + bb[3] - bb[1] + 2],
            fill=(18, 24, 36),
        )
        draw.text((lx, ly), label, fill=TXT_SEC, font=f)


def draw_l_arrow(
    draw: ImageDraw.Draw,
    points: list,
    color: tuple = ARROW_CLR,
    label: str = None,
    label_pos: tuple = None,
) -> None:
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=2)
    px, py = points[-2]
    ex, ey = points[-1]
    arrowhead(draw, ex, ey, math.degrees(math.atan2(ey - py, ex - px)), color=color)
    if label and label_pos:
        f = get_font(12)
        lx, ly = label_pos
        bb = draw.textbbox((0, 0), label, font=f)
        draw.rectangle(
            [lx - 3, ly - 2, lx + bb[2] - bb[0] + 4, ly + bb[3] - bb[1] + 2],
            fill=(18, 24, 36),
        )
        draw.text((lx, ly), label, fill=ACCENT, font=f)


def draw_node(
    img: Image.Image,
    cx: int,
    cy: int,
    nw: int,
    nh: int,
    emoji: str,
    title: str,
    subtitle: str,
    active: bool = False,
) -> None:
    x1, y1, x2, y2 = cx - nw // 2, cy - nh // 2, cx + nw // 2, cy + nh // 2
    if active:
        draw_glow(img, x1, y1, x2, y2)
    draw_rounded_gradient(
        img,
        x1,
        y1,
        x2,
        y2,
        lerp_color(NODE_TOP, (30, 46, 64), 0.5) if active else NODE_TOP,
        NODE_BOT,
        radius=12,
        border_c=ACCENT if active else NODE_BORDER,
        border_w=2 if active else 1,
    )
    d = ImageDraw.Draw(img)
    fe = get_font(20)
    ft = get_font(15, bold=True)
    fs = get_font(12)
    eb = d.textbbox((0, 0), emoji, font=fe)
    d.text(
        (cx - (eb[2] - eb[0]) // 2, y1 + 7),
        emoji,
        fill=ACCENT if active else TXT_PRI,
        font=fe,
    )
    tb = d.textbbox((0, 0), title, font=ft)
    d.text((cx - (tb[2] - tb[0]) // 2, y1 + 34), title, fill=TXT_PRI, font=ft)
    sb = d.textbbox((0, 0), subtitle, font=fs)
    d.text(
        (cx - (sb[2] - sb[0]) // 2, y1 + 56),
        subtitle,
        fill=ACCENT if active else TXT_SEC,
        font=fs,
    )


# ── Architecture diagram ───────────────────────────────────────────────────────


def draw_arch_base(img: Image.Image, active_label: str = None) -> None:
    """Draw complete LangGraph architecture diagram onto img."""
    d = ImageDraw.Draw(img)
    draw_dot_grid(d, W, H)

    # Header bar
    d.rectangle([0, 0, W, 52], fill=(14, 20, 30))
    d.rectangle([0, 52, W, 55], fill=ACCENT)
    f_tl = get_font(24, bold=True)
    ttl = "LangGraph StateGraph  --  Multi-Agent OKR Analytics Flow"
    tb = d.textbbox((0, 0), ttl, font=f_tl)
    d.text(((W - (tb[2] - tb[0])) // 2, 12), ttl, fill=TXT_PRI, font=f_tl)

    # START badge
    sx, sy = 52, 265
    d.ellipse(
        [sx - 22, sy - 22, sx + 22, sy + 22],
        fill=(18, 28, 44),
        outline=ARROW_CLR,
        width=2,
    )
    fs2 = get_font(11, bold=True)
    sb2 = d.textbbox((0, 0), "START", font=fs2)
    d.text(
        (sx - (sb2[2] - sb2[0]) // 2, sy - (sb2[3] - sb2[1]) // 2),
        "START",
        fill=ARROW_CLR,
        font=fs2,
    )

    # Arrows
    draw_h_arrow(d, sx + 22, sy, N["planner"][0] - NW // 2)
    draw_h_arrow(d, N["planner"][0] + NW // 2, 265, N["research"][0] - NW // 2)
    draw_h_arrow(d, N["research"][0] + NW // 2, 265, N["analytics"][0] - NW // 2)
    draw_h_arrow(d, N["analytics"][0] + NW // 2, 265, N["critique"][0] - NW // 2)

    draw_l_arrow(
        d,
        [
            (N["critique"][0], N["critique"][1] + NH // 2),
            (N["critique"][0], N["hitl"][1]),
            (N["hitl"][0] + NW // 2, N["hitl"][1]),
        ],
        color=SUCCESS,
        label="pass >= 0.75",
        label_pos=(
            N["critique"][0] + 6,
            (N["critique"][1] + NH // 2 + N["hitl"][1]) // 2 - 8,
        ),
    )

    draw_l_arrow(
        d,
        [
            (N["hitl"][0] - NW // 2, N["hitl"][1]),
            (480, N["hitl"][1]),
            (480, N["analytics"][1] + NH // 2 + 12),
            (N["analytics"][0] - NW // 2, N["analytics"][1] + NH // 2 + 12),
        ],
        color=DANGER,
        label="revised -> retry",
        label_pos=(486, (N["hitl"][1] + N["analytics"][1]) // 2 - 8),
    )

    draw_v_arrow(
        d,
        N["hitl"][0],
        N["hitl"][1] + NH // 2,
        N["end"][1] - 28,
        color=SUCCESS,
        label="approved",
    )

    # Dashed supervisor connection
    sx0, sy0 = N["analytics"][0] + NW // 2, N["analytics"][1]
    ex0, ey0 = N["supervisor"][0] - 130, N["supervisor"][1]
    total_len = math.hypot(ex0 - sx0, ey0 - sy0)
    steps = int(total_len / 10)
    for i in range(steps):
        if i % 2 == 0:
            t1, t2 = i / steps, (i + 1) / steps
            d.line(
                [
                    (int(sx0 + (ex0 - sx0) * t1), int(sy0 + (ey0 - sy0) * t1)),
                    (int(sx0 + (ex0 - sx0) * t2), int(sy0 + (ey0 - sy0) * t2)),
                ],
                fill=(55, 75, 105),
                width=1,
            )

    # Agent nodes
    nodes_def = [
        ("planner", "P", "Planner", "query -> plan", "Planner Agent"),
        ("research", "R", "Research", "RAG . SEC EDGAR", "Research Agent"),
        ("analytics", "A", "Analytics", "grounded report", "Analytics Agent"),
        ("critique", "C", "Critique", "LLM-as-Judge", "Critique Agent"),
        ("hitl", "H", "HITL", "interrupt() . approve", "HITL Checkpoint"),
        ("supervisor", "S", "Supervisor", "meta-router", "Supervisor Agent"),
    ]
    for key, _e, title, sub, lbl in nodes_def:
        cx, cy = N[key]
        nw_s = 160 if key == "supervisor" else NW
        nh_s = 76 if key == "supervisor" else NH
        draw_node(
            img, cx, cy, nw_s, nh_s, title[:1], title, sub, active=(active_label == lbl)
        )

    # END badge
    ecx, ecy = N["end"]
    draw_rounded_gradient(
        img,
        ecx - 78,
        ecy - 28,
        ecx + 78,
        ecy + 28,
        (20, 52, 30),
        (12, 36, 20),
        radius=28,
        border_c=SUCCESS,
        border_w=2,
    )
    d2 = ImageDraw.Draw(img)
    fe2 = get_font(18, bold=True)
    eb2 = d2.textbbox((0, 0), "END", font=fe2)
    d2.text(
        (ecx - (eb2[2] - eb2[0]) // 2, ecy - (eb2[3] - eb2[1]) // 2),
        "END",
        fill=SUCCESS,
        font=fe2,
    )

    # Floating annotation card
    if active_label and active_label in NODE_ANNOTATIONS:
        lines = NODE_ANNOTATIONS[active_label]
        bx1, by1, bx2, by2 = NODE_BOXES[active_label]
        ann_x = bx2 + 16 if bx2 + 220 < W else bx1 - 226
        ann_y = by1
        f_ann = get_font(13)
        lh, pad, bw = 22, 10, 215
        bh = len(lines) * lh + pad * 2
        draw_rounded_gradient(
            img,
            ann_x,
            ann_y,
            ann_x + bw,
            ann_y + bh,
            (28, 40, 58),
            (18, 26, 40),
            radius=8,
            border_c=ACCENT,
            border_w=1,
        )
        d3 = ImageDraw.Draw(img)
        dot_x = bx2 if ann_x > bx2 else bx1
        d3.line(
            [
                (dot_x, (by1 + by2) // 2),
                (ann_x + (0 if ann_x > bx2 else bw), ann_y + bh // 2),
            ],
            fill=ACCENT,
            width=1,
        )
        d3.ellipse(
            [dot_x - 3, (by1 + by2) // 2 - 3, dot_x + 3, (by1 + by2) // 2 + 3],
            fill=ACCENT,
        )
        for i, line in enumerate(lines):
            d3.text(
                (ann_x + pad, ann_y + pad + i * lh),
                line,
                fill=TXT_PRI if i == 0 else TXT_SEC,
                font=f_ann,
            )

    # Footer strip
    d4 = ImageDraw.Draw(img)
    d4.rectangle([0, H - 30, W, H], fill=CAPTION_BG)
    d4.text(
        (20, H - 20),
        "StateGraph . SqliteSaver . RAGAS 1.0 . DeepEval 100%  .  "
        "github.com/Akash-1512/langgraph-task-orchestrator",
        fill=(45, 65, 90),
        font=get_font(12),
    )


def create_techstack_slide(output_path: str) -> None:
    """Generate premium tech stack card slide."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    draw_dot_grid(d, W, H)
    d.rectangle([0, 0, W, 52], fill=(14, 20, 30))
    d.rectangle([0, 52, W, 55], fill=ACCENT)
    f_t = get_font(28, bold=True)
    ttl = "Production Tech Stack"
    tb = d.textbbox((0, 0), ttl, font=f_t)
    d.text(((W - (tb[2] - tb[0])) // 2, 10), ttl, fill=ACCENT, font=f_t)

    cards = [
        (
            "LangGraph StateGraph",
            "Conditional edges . interrupt() . add_messages reducer",
            "Production-proven orchestration backbone",
        ),
        (
            "Groq LLaMA 3.3 70B",
            "Free 100K tokens/day  ->  Azure OpenAI GPT-4o in prod",
            "Provider-agnostic -- one env var to swap",
        ),
        (
            "ChromaDB -> Qdrant",
            "Local dev: ChromaDB  .  Hosted: Qdrant Cloud 1 GB free",
            "2,284 real SEC EDGAR chunks ingested",
        ),
        (
            "RAGAS + DeepEval",
            "Faithfulness 1.0 . Context Recall 1.0 . GEval 100%",
            "CI/CD gates -- PRs blocked if scores drop",
        ),
        (
            "Langfuse + MLflow",
            "LLM tracing per call . DagsHub experiment tracking",
            "Full observability -- latency, tokens, quality",
        ),
        (
            "FastAPI + Streamlit",
            "WebSocket /ws/run . Docker Compose . Render / Azure",
            "Production-ready API + animated live demo UI",
        ),
    ]
    cw = (W - 120) // 2

    def draw_card(img, x, y, cw, h1, h2, h3):
        ch = 92
        draw_rounded_gradient(
            img,
            x,
            y,
            x + cw,
            y + ch,
            NODE_TOP,
            NODE_BOT,
            radius=10,
            border_c=NODE_BORDER,
            border_w=1,
        )
        dd = ImageDraw.Draw(img)
        dd.rounded_rectangle([x, y, x + 5, y + ch], radius=3, fill=ACCENT)
        dd.text((x + 14, y + 8), h1, fill=TXT_PRI, font=get_font(15, bold=True))
        dd.text((x + 14, y + 36), h2, fill=TXT_SEC, font=get_font(12))
        dd.text((x + 14, y + 58), h3, fill=(70, 150, 210), font=get_font(11))
        return y + ch + 10

    ly = ry = 68
    for c in cards[:3]:
        ly = draw_card(img, 60, ly, cw, *c)
    for c in cards[3:]:
        ry = draw_card(img, 60 + cw + 16, ry, cw, *c)

    d2 = ImageDraw.Draw(img)
    d2.rectangle([0, H - 30, W, H], fill=CAPTION_BG)
    d2.text(
        (20, H - 20),
        "All free-tier providers swap to Azure equivalents via single .env variable  .  DEMO_MODE=true",
        fill=(45, 65, 90),
        font=get_font(13),
    )
    img.save(output_path)
    print(f"   ✅ Tech stack slide: {output_path}")


# ── Frame definitions ──────────────────────────────────────────────────────────

FRAMES = [
    {
        "img": "00a_architecture.png",
        "audio": "00a_narration.mp3",
        "boundaries": "00a_boundaries.json",
        "is_arch": True,
        "caption": "Architecture Overview -- 6-Agent LangGraph StateGraph",
        "script": (
            "This is the Multi-Agent O-K-R Analytics Orchestrator -- "
            "a production-grade agentic system built with LangGraph. "
            "Six specialized agents collaborate in a directed graph. "
            "The Planner Agent decomposes business queries into actionable sub-tasks. "
            "The Research Agent retrieves relevant context from real S-E-C filings. "
            "The Analytics Agent generates structured, grounded reports. "
            "The Critique Agent scores output quality using L-L-M as Judge. "
            "The Human-in-the-Loop checkpoint pauses for human review using interrupt. "
            "And the Supervisor Agent routes between nodes based on state completeness."
        ),
        "highlights": [
            (0.0, 0.063, *NODE_BOXES["Analytics Agent"], "Analytics Agent"),
            (0.063, 0.304, *NODE_BOXES["Planner Agent"], "Planner Agent"),
            (0.304, 0.418, *NODE_BOXES["Research Agent"], "Research Agent"),
            (0.418, 0.633, *NODE_BOXES["Analytics Agent"], "Analytics Agent"),
            (0.633, 0.759, *NODE_BOXES["Critique Agent"], "Critique Agent"),
            (0.759, 0.886, *NODE_BOXES["HITL Checkpoint"], "HITL Checkpoint"),
            (0.886, 1.0, *NODE_BOXES["Supervisor Agent"], "Supervisor Agent"),
        ],
    },
    {
        "img": "00b_techstack.png",
        "audio": "00b_narration.mp3",
        "boundaries": "00b_boundaries.json",
        "is_arch": False,
        "caption": "Tech Stack -- Free Demo . Azure-Ready Production Architecture",
        "script": (
            "The system is built on a fully open, production-grade tech stack. "
            "LangGraph StateGraph provides the orchestration backbone with conditional edges and interrupt support. "
            "Groq's LLaMA 3.3 70 Billion model serves as the free demo L-L-M -- "
            "swappable to Azure OpenAI G-P-T-4-o with a single environment variable. "
            "ChromaDB handles local vector storage, with Qdrant Cloud as the production alternative. "
            "RAGAS and DeepEval gate every pull request -- faithfulness scores 1.0, G-Eval 100 percent pass rate. "
            "Langfuse traces every L-L-M call. MLflow on DagsHub tracks every experiment. "
            "FastAPI with WebSocket streaming powers the backend. Streamlit provides the live demo interface."
        ),
        "highlights": [
            (0.0, 0.14, 44, 62, 644, 162, "LangGraph"),
            (0.14, 0.28, 44, 62, 644, 162, "Groq / Azure OpenAI"),
            (0.28, 0.42, 44, 162, 644, 262, "ChromaDB / Qdrant"),
            (0.42, 0.57, 660, 62, 1262, 162, "RAGAS + DeepEval"),
            (0.57, 0.71, 660, 162, 1262, 262, "Langfuse + MLflow"),
            (0.71, 1.0, 660, 262, 1262, 362, "FastAPI + Streamlit"),
        ],
    },
    {
        "img": "01_loaded.png",
        "audio": "01_narration.mp3",
        "boundaries": "01_boundaries.json",
        "is_arch": False,
        "caption": "Step 1 -- System Overview",
        "script": (
            "Welcome to the Multi-Agent O-K-R Analytics Orchestrator. "
            "This system uses LangGraph to coordinate five specialized AI agents: "
            "Planner, Research, Analytics, Critique, and Human-in-the-Loop. "
            "The knowledge base contains real S-E-C EDGAR filings from Apple, Microsoft, "
            "Google, Tesla, and five other major companies."
        ),
        "highlights": [
            (0.0, 0.4, 460, 290, 820, 470, "5 Agent Nodes"),
            (0.4, 1.0, 78, 618, 760, 658, "Real SEC EDGAR Data"),
        ],
    },
    {
        "img": "02_query_typed.png",
        "audio": "02_narration.mp3",
        "boundaries": "02_boundaries.json",
        "is_arch": False,
        "caption": "Step 2 -- Query Submitted",
        "script": (
            "We submit a real business query: "
            "Analyze Apple's revenue performance from their latest S-E-C 10-K filing. "
            "The Planner Agent will decompose this into specific sub-tasks, "
            "ensuring complete coverage of the analytical objective."
        ),
        "highlights": [
            (0.0, 0.5, 78, 290, 445, 410, "Business Query"),
            (0.5, 1.0, 78, 496, 445, 545, "Run Agent Graph"),
        ],
    },
    {
        "img": "03_running.png",
        "audio": "03_narration.mp3",
        "boundaries": "03_boundaries.json",
        "is_arch": False,
        "caption": "Step 3 -- Agent Pipeline Executing",
        "script": (
            "The agent pipeline is now executing. "
            "The Planner decomposes the query. The Research Agent retrieves "
            "relevant chunks from Apple's 10-K and 10-Q filings using vector similarity search. "
            "The Analytics Agent then generates a structured report grounded "
            "entirely in the retrieved context -- no hallucinations."
        ),
        "highlights": [
            (0.0, 0.159, 460, 355, 522, 465, "Planner"),
            (0.159, 0.273, 460, 355, 522, 465, "Planner"),
            (0.273, 0.636, 535, 355, 598, 465, "Research"),
            (0.636, 1.0, 610, 355, 672, 465, "Analytics"),
        ],
    },
    {
        "img": "04_hitl_ready.png",
        "audio": "04_narration.mp3",
        "boundaries": "04_boundaries.json",
        "is_arch": False,
        "caption": "Step 4 -- HITL Checkpoint: LLM-as-Judge Quality Scores",
        "script": (
            "The Critique Agent has scored the output using L-L-M as Judge. "
            "Faithfulness: 0.90. Coherence: 0.80. Task Completion: 0.60. Overall: 0.78. "
            "The quality gate threshold is 0.75 -- this output has passed. "
            "The graph now pauses at the Human-in-the-Loop checkpoint, "
            "awaiting human review before finalizing the output."
        ),
        "highlights": [
            (0.0, 0.239, 455, 552, 822, 638, "LLM-as-Judge Scores"),
            (0.239, 0.804, 455, 552, 822, 638, "LLM-as-Judge Scores"),
            (0.804, 1.0, 830, 452, 1202, 502, "Approve & Finalize"),
        ],
    },
    {
        "img": "05_approved.png",
        "audio": "05_narration.mp3",
        "boundaries": "05_boundaries.json",
        "is_arch": False,
        "caption": "Step 5 -- Output Approved . Pipeline Complete",
        "script": (
            "The human reviewer has approved the output. "
            "The final analysis -- grounded in real Apple S-E-C filings -- is now complete. "
            "This demonstrates a production-grade agentic system with "
            "quality gates, human oversight, real financial data, "
            "and full observability via Langfuse and MLflow. "
            "Built with LangGraph, Groq, RAGAS, and DeepEval."
        ),
        "highlights": [
            (0.0, 0.082, 455, 492, 820, 548, "Approved Output"),
            (0.082, 0.286, 455, 492, 820, 548, "Approved Output"),
            (0.286, 0.65, 455, 548, 820, 718, "Final Analysis"),
            (0.65, 1.0, 830, 292, 1202, 338, "Pipeline Complete"),
        ],
    },
]


# ── TTS audio generation ───────────────────────────────────────────────────────


async def generate_audio_with_boundaries(frame_cfg: dict) -> list:
    """
    Generate narration audio + word boundary timings for one frame.

    FIX (Flaw 3): Writes to .tmp file first; renames to final path only
    on success. Partial files from network failures are cleaned up.

    FIX (Flaw 2): AudioFileClip opened to read duration is closed immediately.
    """
    audio_path = AUDIO_DIR / frame_cfg["audio"]
    boundary_path = AUDIO_DIR / frame_cfg["boundaries"]

    # Return cached boundaries if valid
    if audio_path.exists() and boundary_path.exists():
        boundaries = json.loads(boundary_path.read_text())
        if boundaries:
            print(f"   ⏭️  {frame_cfg['audio']} -- cached ({len(boundaries)} words)")
            return boundaries
        # Empty cache -- derive proportional timing from existing audio
        ac = AudioFileClip(str(audio_path))
        total_dur = ac.duration
        ac.close()  # FIX Flaw 2: close immediately after reading duration
        words = frame_cfg["script"].split()
        n = len(words)
        boundaries = [
            {"text": w, "offset": (i / n) * total_dur, "duration": total_dur / n}
            for i, w in enumerate(words)
        ]
        boundary_path.write_text(json.dumps(boundaries, indent=2))
        return boundaries

    # Generate new audio via edge-tts
    print(f"   🔊 Generating {frame_cfg['audio']}...")
    communicate = edge_tts.Communicate(frame_cfg["script"], VOICE, rate="+5%")
    boundaries = []
    tmp_path = audio_path.with_suffix(".tmp.mp3")

    try:
        with open(tmp_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    offset = chunk.get("offset", 0)
                    dur = chunk.get("duration", 0)
                    if offset > 1_000_000:  # nanoseconds -> seconds
                        offset /= 10_000_000
                        dur /= 10_000_000
                    boundaries.append(
                        {
                            "text": chunk.get("text", chunk.get("word", "")),
                            "offset": offset,
                            "duration": dur,
                        }
                    )
        tmp_path.rename(audio_path)  # atomic rename on success
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()  # clean up partial file
        raise RuntimeError(f"TTS generation failed for {frame_cfg['audio']}: {e}")

    # Fallback: proportional timing if edge-tts returned no word boundaries
    if not boundaries:
        ac = AudioFileClip(str(audio_path))
        total_dur = ac.duration
        ac.close()  # FIX Flaw 2
        words = frame_cfg["script"].split()
        n = len(words)
        boundaries = [
            {"text": w, "offset": (i / n) * total_dur, "duration": total_dur / n}
            for i, w in enumerate(words)
        ]
        print(f"   ℹ️  Proportional timing applied ({len(boundaries)} words)")

    boundary_path.write_text(json.dumps(boundaries, indent=2))
    print(f"   ✅ {frame_cfg['audio']} ({len(boundaries)} boundaries)")
    return boundaries


async def generate_all_audio() -> list:
    print("🎙️  Generating narrations...")
    return [await generate_audio_with_boundaries(fc) for fc in FRAMES]


# ── Rendering ──────────────────────────────────────────────────────────────────


def build_subtitle_at_time(boundaries: list, t: float) -> str:
    """Return last 14 words spoken up to time t."""
    spoken = [b["text"] for b in boundaries if b["offset"] <= t]
    return " ".join(spoken[-14:]) if spoken else ""


def get_active_highlight(highlights: list, t: float, duration: float):
    """Return (bounding_box, label) for the highlight active at time t."""
    if duration <= 0:
        return None, None
    ratio = min(t / duration, 0.999)
    for entry in highlights:
        if entry[0] <= ratio < entry[1]:
            return (entry[2], entry[3], entry[4], entry[5]), entry[6]
    return None, None


def draw_subtitle(draw: ImageDraw.Draw, text: str, frame_w: int, frame_h: int) -> None:
    """Draw subtitle bar with yellow left accent above the caption area."""
    if not text:
        return
    font = get_font(20)
    lines, cur = [], ""
    for word in text.split():
        test = (cur + " " + word).strip()
        if draw.textbbox((0, 0), test, font=font)[2] > frame_w - 100:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)
    lines = lines[-2:]
    lh = 32
    total_h = len(lines) * lh + 18
    sy = frame_h - total_h - 22
    draw.rectangle([28, sy - 8, frame_w - 28, sy + total_h + 4], fill=(0, 0, 0))
    draw.rectangle([28, sy - 8, 36, sy + total_h + 4], fill=ACCENT)
    for i, line in enumerate(lines):
        lb = draw.textbbox((0, 0), line, font=font)
        lx = (frame_w - (lb[2] - lb[0])) // 2
        ly = sy + i * lh + 9
        draw.text((lx + 1, ly + 1), line, fill=(15, 15, 15), font=font)
        draw.text((lx, ly), line, fill=(255, 255, 255), font=font)


def make_canvas(img: Image.Image, caption: str, subtitle: str) -> Image.Image:
    """Attach caption bar + subtitle to a rendered frame."""
    canvas = Image.new("RGB", (W, H + CAPTION_HEIGHT), BG)
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, H, W, H + CAPTION_HEIGHT], fill=CAPTION_BG)
    d.rectangle([0, H, W, H + 2], fill=ACCENT)
    f_cap = get_font(16, bold=True)
    cb = d.textbbox((0, 0), caption, font=f_cap)
    d.text(
        ((W - (cb[2] - cb[0])) // 2, H + (CAPTION_HEIGHT - (cb[3] - cb[1])) // 2),
        caption,
        fill=TXT_SEC,
        font=f_cap,
    )
    draw_subtitle(d, subtitle, W, H)
    return canvas


def render_arch(
    caption: str, subtitle: str, highlights: list, t: float, duration: float
) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    _, active_label = get_active_highlight(highlights, t, duration)
    draw_arch_base(img, active_label)
    return np.array(make_canvas(img, caption, subtitle))


def render_standard(
    img_path: str,
    caption: str,
    subtitle: str,
    highlights: list,
    t: float,
    duration: float,
) -> np.ndarray:
    img = Image.open(img_path).convert("RGBA")
    img = img.resize((W, H), Image.LANCZOS)
    box, label = get_active_highlight(highlights, t, duration)
    if box:
        x1, y1, x2, y2 = box
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dov = ImageDraw.Draw(ov)
        dov.rectangle([x1, y1, x2, y2], fill=HIGHLIGHT_FILL)
        for i in range(3):
            dov.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=ACCENT + (220,))
        f_lbl = get_font(13, bold=True)
        lb = dov.textbbox((0, 0), label, font=f_lbl)
        lw, lh = lb[2] - lb[0] + 12, lb[3] - lb[1] + 6
        lx, ly = x1, max(0, y1 - lh - 3)
        dov.rectangle([lx, ly, lx + lw, ly + lh], fill=(255, 213, 0, 230))
        dov.text((lx + 6, ly + 3), label, fill=(0, 0, 0, 255), font=f_lbl)
        img = Image.alpha_composite(img, ov)
    img = img.convert("RGB")
    return np.array(make_canvas(img, caption, subtitle))


def build_clip(frame_cfg: dict, boundaries: list) -> VideoClip:
    """Build one MoviePy VideoClip for a frame."""
    img_path = str(SCREENSHOTS_DIR / frame_cfg["img"])
    audio_path = str(AUDIO_DIR / frame_cfg["audio"])
    caption = frame_cfg["caption"]
    highlights = frame_cfg["highlights"]
    is_arch = frame_cfg.get("is_arch", False)

    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration + 0.6
    # Note: do NOT call audio_clip.close() here -- MoviePy needs it open during render.

    if is_arch:

        def make_frame(t):
            sub = build_subtitle_at_time(boundaries, t)
            return render_arch(caption, sub, highlights, t, duration)

    else:

        def make_frame(t):
            sub = build_subtitle_at_time(boundaries, t)
            return render_standard(img_path, caption, sub, highlights, t, duration)

    clip = VideoClip(make_frame, duration=duration)
    clip = clip.with_audio(audio_clip)
    return clip


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("🖼️  Generating slides...")
    # Save static architecture PNG
    img_arch = Image.new("RGB", (W, H), BG)
    draw_arch_base(img_arch)
    img_arch.save(str(SCREENSHOTS_DIR / "00a_architecture.png"))
    print("   ✅ Architecture diagram saved")
    create_techstack_slide(str(SCREENSHOTS_DIR / "00b_techstack.png"))

    all_boundaries = asyncio.run(generate_all_audio())

    print("\n🎬 Building premium video clips...")
    clips = []
    for i, (fc, bnd) in enumerate(zip(FRAMES, all_boundaries)):
        print(f"   🎞️  Clip {i+1}/{len(FRAMES)}: {fc['img']}")
        clips.append(build_clip(fc, bnd))

    print("\n⏳ Rendering final MP4...")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        str(OUTPUT_VIDEO), fps=FPS, codec="libx264", audio_codec="aac", logger="bar"
    )
    print(f"\n✅  {OUTPUT_VIDEO}  (~{final.duration:.0f}s)  Voice: {VOICE}")


if __name__ == "__main__":
    main()
