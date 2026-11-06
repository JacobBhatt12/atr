"""Draws assets/demo.gif: a chatbot using ATR (numbers are from real runs, see benchmarks/).

    python assets/make_chat_demo.py        # needs Pillow; uses macOS fonts
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1100, 700
BG, PANEL, BAR = "#0e1116", "#161b22", "#1c222b"
TXT, DIM, USER = "#e6edf3", "#8b949e", "#1f6feb"
GREEN, RED, BLUE, AMBER = "#3fb950", "#f85149", "#58a6ff", "#d29922"
UI = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"


def font(path, size):
    return ImageFont.truetype(path, size)


F, FB, FM, FS, FT = font(UI, 22), font(UI, 22), font(MONO, 18), font(UI, 17), font(UI, 36)
frames: list[tuple[Image.Image, int]] = []


def base(title="Any chatbot  +  ATR connector") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 54), fill=BAR)
    for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        d.ellipse((22 + i * 26, 19, 36 + i * 26, 33), fill=c)
    d.text((W / 2, 27), title, font=FS, fill=DIM, anchor="mm")
    return im, d


def wrap(d, text, fnt, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur]


def user_bubble(d, y, text):
    lines = wrap(d, text, F, 640)
    w = max(d.textlength(t, font=F) for t in lines) + 40
    h = len(lines) * 30 + 22
    d.rounded_rectangle((W - 40 - w, y, W - 40, y + h), 18, fill=USER)
    for i, t in enumerate(lines):
        d.text((W - 20 - w, y + 11 + i * 30), t, font=F, fill="white")
    return y + h + 18


def chip(d, y, call, result=None, result_color=DIM):
    d.ellipse((44, y + 8, 54, y + 18), fill=BLUE)
    d.text((66, y), call, font=FM, fill=BLUE)
    y += 28
    if result:
        d.text((66, y), result, font=FM, fill=result_color)
        y += 28
    return y + 10


def assistant(d, y, lines):
    h = len(lines) * 30 + 22
    d.rounded_rectangle((40, y, 40 + 820, y + h), 18, fill=PANEL)
    for i, t in enumerate(lines):
        d.text((62, y + 11 + i * 30), t, font=F, fill=TXT)
    return y + h + 14


def meter(d, label_a, a, label_b, b, y=598, note=""):
    d.rectangle((0, y - 14, W, H), fill=BAR)
    mx = 470
    d.text((40, y), label_a, font=FS, fill=DIM)
    d.rectangle((250, y + 2, 250 + mx, y + 22), fill=RED)
    d.text((250 + mx + 12, y), a, font=FS, fill=RED)
    d.text((40, y + 36), label_b, font=FS, fill=DIM)
    d.rectangle((250, y + 38, 250 + max(6, int(mx * 0.008)), y + 58), fill=GREEN)
    d.text((250 + 6 + 12, y + 36), b, font=FS, fill=GREEN)
    if note:
        d.multiline_text((W - 40, y + 28), note, font=FB, fill=GREEN, anchor="rm", align="right", spacing=2)


def add(im, ms):
    frames.append((im, ms))


def type_scene(question, hold=0):
    for n in range(0, len(question) + 1, 4):
        im, d = base()
        user_bubble(d, 80, question[:n] + ("▌" if n < len(question) else ""))
        add(im, 60)
    return


def scene_github():
    q = "Which open issues in modelcontextprotocol/python-sdk are about OAuth? Use atr."
    type_scene(q)
    steps = [
        ('atr_find_tools("github list issues")', "→ github__list_issues  [read-only]", DIM),
        ('atr_read(github__list_issues, question="which are about OAuth?")', "30 issues · 53,724 → 419 tokens", GREEN),
    ]
    for k in range(1, 3):
        im, d = base()
        y = user_bubble(d, 80, q)
        for call, res, col in steps[:k]:
            y = chip(d, y, call, res, col)
        add(im, 1400)
    answer = ["4 open issues are about OAuth:",
              "#3781  OAuth scope selection falls back to scopes_supported",
              "#3778  Advertise scopes_supported separately from required_scopes",
              "#3769  No error should be added for a request with no token",
              "#3768  Issuers compared after URL normalisation (CIMD client_id)"]
    for shown in (1, 5):
        im, d = base()
        y = user_bubble(d, 80, q)
        for call, res, col in steps:
            y = chip(d, y, call, res, col)
        assistant(d, y, answer[:shown])
        meter(d, "Without ATR", "53,724 tokens", "With ATR", "419 tokens", note="99%\nfewer")
        add(im, 700 if shown == 1 else 3200)


def scene(question, steps, answer, meter_args=None):
    """steps: [(call, result, color)]; the chat shows each step, then the answer (and a token meter)."""
    type_scene(question)
    for k in range(1, len(steps) + 1):
        im, d = base()
        y = user_bubble(d, 80, question)
        for call, res, col in steps[:k]:
            y = chip(d, y, call, res, col)
        add(im, 1300)
    im, d = base()
    y = user_bubble(d, 80, question)
    for call, res, col in steps:
        y = chip(d, y, call, res, col)
    assistant(d, y, answer)
    if meter_args:
        meter(d, *meter_args[:4], note=meter_args[4])
    add(im, 3300)


def scene_setup():
    im, d = base()
    d.text((W / 2, 120), "How to get your chatbot to use ATR", font=FT, fill=TXT, anchor="mm")
    d.text((W / 2, 175), "Chatbots reach for their own web tools first. Two ways to pick ATR:", font=F, fill=DIM, anchor="mm")
    d.text((100, 235), "Add the connector once:  Settings → Connectors → Add custom connector", font=F, fill=TXT)
    d.text((130, 275), "https://atr-mcp.vercel.app/mcp   (no sign-in)", font=FM, fill=BLUE)
    d.text((100, 335), "A.  Just add \u201cuse atr\u201d to your question", font=FB, fill=GREEN)
    d.rounded_rectangle((130, 380, W - 130, 436), 14, fill=USER)
    d.text((155, 392), "What\u2019s the latest on the MCP spec? Use atr.", font=F, fill="white")
    d.text((100, 480), "B.  Or set it once, so you never have to:  Profile → personal preferences", font=FB, fill=GREEN)
    d.rounded_rectangle((130, 525, W - 130, 625), 14, fill=PANEL, outline=BLUE)
    for k, t in enumerate(wrap(d, "When I ask you to search the web or read a link, use atr_search and atr_fetch instead of the built-in web tools.", FS, W - 330)):
        d.text((155, 540 + k * 28), t, font=FS, fill=TXT)
    add(im, 7000)


def scene_more():
    scene("When was Python 3.14 released? Use atr to search.",
          [('atr_search("python 3.14 release date")', "5 results from DuckDuckGo + Bing, merged · ~280 tokens", GREEN)],
          ["Python 3.14.0 was released on October 7, 2025.", "Source: python.org/downloads/release/python-3140"])
    scene("In the Attention Is All You Need paper, what BLEU did the big Transformer get? Use atr.",
          [('atr_fetch("arxiv.org/pdf/1706.03762", question="BLEU big Transformer")', "9,962 → 263 tokens (just the sentences that answer)", GREEN)],
          ["English-to-German: 28.4 BLEU (new state of the art).", "English-to-French: 41.0 BLEU, at under 1/4 the training cost."],
          ("Whole PDF", "9,962 tokens", "With ATR", "263 tokens", "97%\nfewer"))
    scene("Find coffee shops in Tampa that are open Sunday morning. Use atr.",
          [('atr_places("coffee shops", location="Tampa, FL")', "3 OpenStreetMap places with hours, phone and map link · ~260 tokens", GREEN)],
          ["Cafe at 1920 (Ybor City), Sat-Sun 9:00-15:00", "La Creperia Cafe (Ybor City), Sat-Sun 9:00-16:00", "Capital One Cafe (Davis Islands), Sun 9:00-17:00"])
    scene("What's the weather in Tampa this weekend? Use atr.",
          [('atr_weather("Tampa, FL", days=3)', "current conditions + 3-day forecast · ~110 tokens", GREEN)],
          ["Now: clear, 79°F. Sat: overcast, high 78°F, 5% rain.", "Sun: overcast, high 82°F, low 60°F, no rain."])


def scene_end():
    im, d = base()
    d.text((W / 2, 150), "ATR", font=font(UI, 84), fill=TXT, anchor="mm")
    d.text((W / 2, 225), "Read links and PDFs, search the web, GitHub, places, weather.", font=F, fill=DIM, anchor="mm")
    d.text((W / 2, 258), "A free connector that makes your chatbot use fewer tokens.", font=F, fill=DIM, anchor="mm")
    rows = [("GitHub tool list", "32,563", "917"), ("30 GitHub issues", "53,724", "419"),
            ("8 web lookups (pages, PDF, search)", "65,252", "5,636")]
    for i, (a, b, c) in enumerate(rows):
        y = 320 + i * 58
        d.text((170, y), a, font=F, fill=TXT)
        d.text((760, y), b, font=F, fill=RED, anchor="ra")
        d.text((790, y), "→", font=F, fill=DIM)
        d.text((840, y), c, font=FB, fill=GREEN)
    d.text((W / 2, 540), "Add  https://atr-mcp.vercel.app/mcp  as a connector", font=font(MONO, 24), fill=BLUE, anchor="mm")
    d.text((W / 2, 600), "atr-mcp.vercel.app   ·   free, no sign-up", font=F, fill=DIM, anchor="mm")
    add(im, 4500)


scene_setup()
scene_github()
scene_more()
scene_end()
out = Path(__file__).with_name("demo.gif")
pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im, _ in frames]
pal[0].save(out, save_all=True, append_images=pal[1:], duration=[ms for _, ms in frames], loop=0, optimize=True, disposal=1)
print(out, out.stat().st_size // 1024, "KB,", len(frames), "frames")
