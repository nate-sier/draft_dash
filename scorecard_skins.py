# VERSION: option1_original_card_grid_60th_capacity_v61
"""Portable Nationals/defensive scorecard skins.

Use this module inside another app by passing the existing scorecard data into
``render_scorecard_option_1`` or ``render_scorecard_option_2``.

Both functions can either write a PDF to ``output_path`` or return PDF bytes
when ``output_path`` is omitted.
"""

from __future__ import annotations

import os
import platform
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = letter

# Official Washington Nationals colors.
NATS_NAVY = colors.HexColor("#14225A")
NATS_RED = colors.HexColor("#AB0003")
NATS_WHITE = colors.HexColor("#FFFFFF")

# Neutral support colors.
PAGE_BG = colors.HexColor("#F5F6F9")
TEXT_DARK = colors.HexColor("#1A1A1A")
LINE_GRAY = colors.HexColor("#D6DAE2")
TRACK_GRAY = colors.HexColor("#E8ECF2")
GRID_GRAY = colors.HexColor("#D0D6DF")
MUTED_TEXT = colors.HexColor("#6F7785")

# Defensive report percentile palette.
PCTL_GREEN_3 = colors.HexColor("#A8D98C")
PCTL_GREEN_2 = colors.HexColor("#CDEBB8")
PCTL_GREEN_1 = colors.HexColor("#E2F1D8")
PCTL_RED_1 = colors.HexColor("#FFE9EC")
PCTL_RED_2 = colors.HexColor("#F7C3CB")
PCTL_RED_3 = colors.HexColor("#F09AA5")

_FONT_REG = "GillSans"
_FONT_BOLD = "GillSans-Bold"
_REGISTERED = False
_USE_GILL = False


def _register_fonts() -> None:
    global _REGISTERED, _USE_GILL
    if _REGISTERED:
        return
    _REGISTERED = True
    mac_path = "/System/Library/Fonts/Supplemental/GillSans.ttc"
    try:
        if platform.system() == "Darwin" and os.path.isfile(mac_path):
            pdfmetrics.registerFont(TTFont(_FONT_REG, mac_path, subfontIndex=0))
            pdfmetrics.registerFont(TTFont(_FONT_BOLD, mac_path, subfontIndex=1))
            _USE_GILL = True
    except Exception:
        _USE_GILL = False


def _font_name(variant: str = "regular") -> str:
    _register_fonts()
    if _USE_GILL:
        return _FONT_BOLD if variant == "bold" else _FONT_REG
    return "Helvetica-Bold" if variant == "bold" else "Helvetica"


def _set_font(c: canvas.Canvas, variant: str, size: float) -> None:
    c.setFont(_font_name(variant), size)


def _text_width(text: Any, variant: str, size: float) -> float:
    return pdfmetrics.stringWidth(str(text), _font_name(variant), size)


def _fit_text(
    c: canvas.Canvas,
    text: Any,
    x: float,
    y: float,
    max_width: float,
    *,
    align: str = "left",
    variant: str = "bold",
    max_size: float = 12,
    min_size: float = 6,
    fill: colors.Color = TEXT_DARK,
) -> None:
    value = str(text)
    size = max_size
    while size > min_size and _text_width(value, variant, size) > max_width:
        size -= 0.25
    c.setFillColor(fill)
    _set_font(c, variant, size)
    if align == "right":
        c.drawRightString(x, y, value)
    elif align == "center":
        c.drawCentredString(x, y, value)
    else:
        c.drawString(x, y, value)


def _pctl_label(percentile: int | float | None) -> str:
    if percentile is None:
        return ""
    return f"{int(round(float(percentile)))} pctl"


def _pctl_fill(percentile: int | float | None) -> colors.Color:
    if percentile is None:
        return NATS_WHITE
    p = int(max(0, min(100, round(float(percentile)))))
    if p >= 86:
        return PCTL_GREEN_3
    if p >= 71:
        return PCTL_GREEN_2
    if p >= 56:
        return PCTL_GREEN_1
    if p >= 45:
        return NATS_WHITE
    if p >= 30:
        return PCTL_RED_1
    if p >= 15:
        return PCTL_RED_2
    return PCTL_RED_3


def _draw_logo(
    c: canvas.Canvas, path: str | None, x: float, y: float, w: float, h: float
) -> None:
    if path and Path(path).exists():
        c.drawImage(
            ImageReader(path),
            x,
            y,
            w,
            h,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )


def _finish_pdf(buffer: BytesIO, output_path: str | Path | None) -> bytes:
    pdf_bytes = buffer.getvalue()
    if output_path is not None:
        Path(output_path).write_bytes(pdf_bytes)
    return pdf_bytes


def _make_canvas() -> tuple[canvas.Canvas, BytesIO]:
    buffer = BytesIO()
    return canvas.Canvas(buffer, pagesize=letter), buffer


def _draw_circle_image(
    c: canvas.Canvas, path: str | None, cx: float, cy: float, r: float, initials: str = ""
) -> None:
    c.saveState()
    c.setFillColor(colors.HexColor("#BEBFC3"))
    c.setStrokeColor(colors.HexColor("#BDBDBD"))
    c.setLineWidth(1)
    c.circle(cx, cy, r, stroke=1, fill=1)
    if path and Path(path).exists():
        clip = c.beginPath()
        clip.circle(cx, cy, r)
        c.clipPath(clip, stroke=0, fill=0)
        c.drawImage(
            ImageReader(path),
            cx - r,
            cy - r,
            2 * r,
            2 * r,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )
    c.restoreState()
    if not path and initials:
        _fit_text(c, initials, cx, cy - 5, r * 1.35, align="center", max_size=16, min_size=10)


def _draw_option_1_outer_frame(c: canvas.Canvas) -> None:
    margin = 18
    c.setFillColor(NATS_WHITE)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    c.setStrokeColor(colors.black)
    c.setLineWidth(2)
    c.rect(margin, margin, PAGE_W - 2 * margin, PAGE_H - 2 * margin, stroke=1, fill=0)


def _draw_option_1_header(c: canvas.Canvas, data: dict[str, Any]) -> float:
    margin = 18
    header_h = 66
    inset = 10
    left = margin + inset
    right = PAGE_W - margin - inset
    top = PAGE_H - margin - inset
    bottom = top - header_h
    mid_y = bottom + header_h * 0.52

    player_name = data.get("player_name", "Player")
    head_cx = left + 34
    head_r = 28
    initials = "".join(part[:1] for part in str(player_name).split()[:2]).upper()
    _draw_circle_image(c, data.get("headshot_path"), head_cx, mid_y, head_r, initials=initials)

    logo_w, logo_h = 58, 40
    logo_x = right - 36 - logo_w / 2
    logo_y = bottom + header_h - 4 - logo_h
    _draw_logo(c, data.get("logo_path"), logo_x, logo_y, logo_w, logo_h)

    center_left = head_cx + head_r + 14
    center_right = logo_x - 14
    center_x = (center_left + center_right) / 2
    center_w = max(120, center_right - center_left)

    _fit_text(
        c,
        f"{str(player_name).upper()}, {str(data.get('position', '')).upper()}",
        center_x,
        bottom + header_h - 22,
        center_w,
        align="center",
        max_size=14,
        min_size=9,
    )
    _fit_text(
        c,
        data.get("context", ""),
        center_x,
        bottom + header_h - 42,
        center_w,
        align="center",
        max_size=9,
        min_size=6,
    )
    _fit_text(
        c,
        data.get("report_date", ""),
        right - 6,
        bottom + 5,
        80,
        align="right",
        max_size=11,
        min_size=7,
    )
    return bottom - 10


def _draw_option_1_metric_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    value: str,
    percentile: int | float | None = None,
) -> None:
    """Option 1 top-card renderer using original Skin 1 organization.

    Athlete Group and Program Focus stay in the same card grid as the original
    skin, but their value text is manually wrapped and reduced so it fits.
    """
    label_clean = str(label).strip()
    value_clean = str(value).strip()
    is_long_bottom_card = label_clean.lower() in {"athlete group", "program focus"}
    is_60th_capacity_card = label_clean.lower() == "60th percentile capacity"

    # Keep the original feel while giving long-value cards a little more room.
    # The 60th-percentile card is a two-line label with a compact score range.
    label_w = w * (0.60 if is_60th_capacity_card else (0.44 if is_long_bottom_card else 0.54))

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)

    c.setFillColor(NATS_NAVY)
    c.rect(x, y, label_w, h, stroke=1, fill=1)

    c.setFillColor(_pctl_fill(percentile))
    c.rect(x + label_w, y, w - label_w, h, stroke=1, fill=1)

    if is_60th_capacity_card:
        c.setFillColor(NATS_WHITE)
        _set_font(c, "bold", 5.8)
        c.drawCentredString(x + label_w / 2, y + h / 2 + 2.0, "60th Percentile")
        c.drawCentredString(x + label_w / 2, y + h / 2 - 5.0, "Capacity")
    else:
        _fit_text(
            c,
            label_clean,
            x + label_w / 2,
            y + h / 2 - 3,
            label_w - 6,
            align="center",
            max_size=8.0 if is_long_bottom_card else 9,
            min_size=4.7,
            fill=NATS_WHITE,
        )

    value_x = x + label_w
    value_w = w - label_w

    if is_long_bottom_card:
        manual_lines = {
            "low ci / foundational": ["Low CI /", "Foundational"],
            "foundational strength/capacity": ["Foundational", "Strength/Capacity"],
            "high ci - low p1": ["High CI -", "Low P1"],
            "high ci - high p1": ["High CI -", "High P1"],
            "p1 development": ["P1", "Development"],
            "advanced": ["Advanced"],
            "unclassified": ["Unclassified"],
        }
        lines = manual_lines.get(value_clean.lower())
        if lines is None:
            words, lines, cur = value_clean.split(), [], ""
            for word in words:
                test = word if not cur else f"{cur} {word}"
                if len(test) <= 18:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
                if len(lines) == 2:
                    break
            if cur and len(lines) < 2:
                lines.append(cur)
            lines = lines[:2] or ["-"]

        c.setFillColor(TEXT_DARK)
        # Compact fixed text so the full values fit in the original Skin 1 card grid.
        _set_font(c, "bold", 6.6)
        if len(lines) == 1:
            c.drawCentredString(value_x + value_w / 2, y + h / 2 - 2.5, lines[0])
        else:
            c.drawCentredString(value_x + value_w / 2, y + h / 2 + 3.2, lines[0])
            c.drawCentredString(value_x + value_w / 2, y + h / 2 - 7.0, lines[1])
        return

    _fit_text(
        c,
        value_clean,
        value_x + value_w / 2,
        y + h / 2 - 3,
        value_w - 8,
        align="center",
        max_size=10 if is_60th_capacity_card else 11,
        min_size=5.3 if is_60th_capacity_card else 6,
    )


def _draw_option_1_summary_cards(c: canvas.Canvas, y_top: float, cards: list[dict[str, Any]]) -> float:
    """Draw Option 1 score summary using the original Skin 1 card grid.

    This preserves the original three-column scorecard grid:
    - 3 equal-width cards on the first row
    - up to 3 equal-width cards on the second row

    The sixth card is used for the 60th Percentile Capacity projection and
    naturally sits directly beneath Potential to Gain.
    """
    x = 46
    total_w = PAGE_W - 92
    gap = 9
    cols = 3
    card_w = (total_w - gap * (cols - 1)) / cols
    card_h = 28

    for i, card in enumerate(cards):
        col = i % cols
        row = i // cols
        cx = x + col * (card_w + gap)
        cy = y_top - card_h - row * (card_h + 8)
        _draw_option_1_metric_card(
            c,
            cx,
            cy,
            card_w,
            card_h,
            label=str(card["label"]),
            value=str(card.get("value", "-")),
            percentile=card.get("percentile"),
        )

    rows = (len(cards) + cols - 1) // cols
    return y_top - rows * (card_h + 8) - 6


def _draw_option_1_panel(
    c: canvas.Canvas, x: float, y_top: float, w: float, h: float, title: str
) -> tuple[float, float, float, float]:
    y = y_top - h
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.rect(x, y, w, h, stroke=1, fill=0)
    _fit_text(c, title, x + w / 2, y_top - 20, w - 20, align="center", max_size=15, min_size=9)
    return x + 10, y + 10, w - 20, h - 36


def _draw_option_1_metric_row(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    *,
    label: str,
    value: str,
    percentile: int | float | None,
) -> None:
    # A missing percentile must remain visibly unscored. In particular, CI-100ms
    # is optional in some exports and should not be rendered as a fake 50th-percentile value.
    try:
        p = int(max(0, min(100, round(float(percentile)))))
        has_percentile = True
    except (TypeError, ValueError):
        p = None
        has_percentile = False

    label_w = 76
    value_w = 62
    track_x = x + label_w + 8
    track_w = w - label_w - value_w - 18
    track_h = 12
    track_y = y + 8

    _fit_text(
        c,
        label,
        x + label_w,
        y + 9,
        label_w,
        align="right",
        variant="bold",
        max_size=8.5,
        min_size=5,
    )
    c.setFillColor(colors.HexColor("#F1F5F9"))
    c.setStrokeColor(colors.HexColor("#B8C0CC"))
    c.setLineWidth(0.4)
    c.rect(track_x, track_y, track_w, track_h, stroke=1, fill=1)
    c.setStrokeColor(colors.HexColor("#59616D"))
    c.setLineWidth(0.5)
    c.line(track_x + track_w / 2, track_y - 3, track_x + track_w / 2, track_y + track_h + 3)

    if has_percentile:
        c.setFillColor(_pctl_fill(percentile))
        c.rect(track_x, track_y, track_w * p / 100.0, track_h, stroke=0, fill=1)

        marker_x = track_x + track_w * p / 100.0
        c.setFillColor(NATS_WHITE)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.8)
        c.circle(marker_x, track_y + track_h / 2, 3.6, stroke=1, fill=1)
        _fit_text(c, _pctl_label(percentile), marker_x, track_y - 10, 42, align="center", max_size=6.2, min_size=4.8)

    _fit_text(c, value, x + w, y + 9, value_w, align="right", max_size=8, min_size=5)
    c.setStrokeColor(colors.HexColor("#E5E7EB"))
    c.setLineWidth(0.3)
    c.line(x, y - 5, x + w, y - 5)


def _draw_option_1_rows_component(
    c: canvas.Canvas, *, title: str, y_top: float, rows: list[dict[str, Any]]
) -> float:
    x = 46
    w = PAGE_W - 92
    h = max(82, 37 + len(rows) * 27)
    inner_x, _, inner_w, _ = _draw_option_1_panel(c, x, y_top, w, h, title)
    row_y = y_top - 50
    for row in rows:
        _draw_option_1_metric_row(
            c,
            inner_x,
            row_y,
            inner_w,
            label=str(row["label"]),
            value=str(row.get("value", "-")),
            percentile=row.get("percentile"),
        )
        row_y -= 27
    return y_top - h - 14


def render_scorecard_option_1(
    data: dict[str, Any], output_path: str | Path | None = None
) -> bytes:
    """Render the more defensive-report styled scorecard skin."""
    c, buffer = _make_canvas()
    _draw_option_1_outer_frame(c)
    y = _draw_option_1_header(c, data)
    y = _draw_option_1_summary_cards(c, y, data.get("summary_cards", []))

    for section in data.get("sections", []):
        needed_h = max(82, 37 + len(section.get("rows", [])) * 27)
        if y - needed_h < 36:
            c.showPage()
            _draw_option_1_outer_frame(c)
            y = PAGE_H - 42
        y = _draw_option_1_rows_component(
            c,
            title=str(section.get("title", "")),
            y_top=y,
            rows=section.get("rows", []),
        )

    c.showPage()
    c.save()
    return _finish_pdf(buffer, output_path)


def _draw_option_2_page_background(c: canvas.Canvas) -> None:
    c.setFillColor(PAGE_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)


def _draw_option_2_header(c: canvas.Canvas, data: dict[str, Any]) -> float:
    x = 42
    y_top = PAGE_H - 32
    w = PAGE_W - 84
    h = 98
    y = y_top - h

    c.setFillColor(NATS_NAVY)
    c.rect(x, y, w, h, stroke=0, fill=1)
    c.setFillColor(NATS_RED)
    c.rect(x, y_top - 6, w, 6, stroke=0, fill=1)
    _fit_text(
        c,
        data.get("banner_label", "WASHINGTON NATIONALS - DRAFT SCOUTING"),
        x + 24,
        y_top - 27,
        w - 190,
        max_size=8.5,
        min_size=6,
        fill=NATS_WHITE,
    )
    _fit_text(
        c,
        data.get("player_name", "Player"),
        x + 24,
        y_top - 61,
        w - 200,
        max_size=25,
        min_size=15,
        fill=NATS_WHITE,
    )
    _set_font(c, "regular", 9.5)
    c.setFillColor(colors.HexColor("#DDE3F0"))
    c.drawString(x + 24, y + 15, data.get("subtitle", data.get("context", "")))
    _draw_logo(c, data.get("logo_path"), x + w - 226, y + 26, 54, 38)

    score_w = 72
    score_h = 68
    score_x = x + w - 112
    score_y = y + 15
    c.setFillColor(NATS_WHITE)
    c.rect(score_x, score_y, score_w, score_h, stroke=0, fill=1)
    _fit_text(
        c,
        data.get("capacity", "-"),
        score_x + score_w / 2,
        score_y + 38,
        score_w - 12,
        align="center",
        max_size=24,
        min_size=15,
        fill=NATS_RED,
    )
    _fit_text(
        c,
        "CAPACITY",
        score_x + score_w / 2,
        score_y + 12,
        score_w - 12,
        align="center",
        max_size=6.2,
        min_size=5,
        fill=MUTED_TEXT,
    )
    return y - 16


def _draw_option_2_summary_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    value: str,
    top_color: colors.Color,
    filled: bool = False,
) -> None:
    c.setFillColor(NATS_WHITE if not filled else NATS_NAVY)
    c.rect(x, y, w, h, stroke=0, fill=1)
    c.setStrokeColor(LINE_GRAY)
    c.setLineWidth(0.75)
    c.rect(x, y, w, h, stroke=1, fill=0)
    c.setFillColor(top_color)
    c.rect(x, y + h - 5, w, 5, stroke=0, fill=1)
    _fit_text(
        c,
        label.upper(),
        x + 10,
        y + h - 18,
        w - 18,
        max_size=7.3,
        min_size=5.5,
        fill=NATS_WHITE if filled else MUTED_TEXT,
    )
    _fit_text(
        c,
        value,
        x + 10,
        y + 10,
        w - 18,
        max_size=14,
        min_size=9,
        fill=NATS_WHITE if filled else NATS_NAVY,
    )


def _draw_option_2_summary_grid(c: canvas.Canvas, y_top: float, cards: list[dict[str, Any]]) -> float:
    left = 42
    w = PAGE_W - 84
    gap = 14
    card_h = 50
    top_w = (w - 2 * gap) / 3
    y1 = y_top - card_h
    for i, card in enumerate(cards[:3]):
        top_color = NATS_RED if i == 0 else NATS_NAVY
        if str(card.get("label", "")).lower() in {"ci tier", "program"}:
            top_color = NATS_RED
        _draw_option_2_summary_card(
            c,
            left + i * (top_w + gap),
            y1,
            top_w,
            card_h,
            label=str(card["label"]),
            value=str(card.get("value", "-")),
            top_color=card.get("top_color", top_color),
            filled=bool(card.get("filled", i == 0)),
        )
    bottom_w = top_w
    y2 = y1 - card_h - 8
    start = left + top_w * 0.52
    for i, card in enumerate(cards[3:5]):
        _draw_option_2_summary_card(
            c,
            start + i * (bottom_w + gap),
            y2,
            bottom_w,
            card_h,
            label=str(card["label"]),
            value=str(card.get("value", "-")),
            top_color=card.get("top_color", NATS_RED),
            filled=bool(card.get("filled", False)),
        )
    return y2 - 12


def _draw_option_2_section_header(c: canvas.Canvas, x: float, y_top: float, title: str, w: float) -> None:
    _fit_text(c, title, x + 8, y_top - 16, w - 16, max_size=16, min_size=11, fill=NATS_NAVY)
    c.setFillColor(NATS_RED)
    c.rect(x + 8, y_top - 31, 46, 4, stroke=0, fill=1)
    c.setStrokeColor(LINE_GRAY)
    c.setLineWidth(0.5)
    c.line(x + 62, y_top - 29, x + w - 10, y_top - 29)


def _draw_option_2_metric_row(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    *,
    label: str,
    score: int | float | None,
    value: str,
) -> None:
    label_w = 88
    value_w = 74
    bar_x = x + label_w + 8
    bar_w = w - label_w - value_w - 24
    bar_h = 15
    score_num = 0 if score is None else int(max(0, min(100, round(float(score)))))

    _fit_text(c, label, x + label_w, y + 4, label_w - 4, align="right", variant="regular", max_size=8.0, min_size=5)
    c.setFillColor(TRACK_GRAY)
    c.rect(bar_x, y, bar_w, bar_h, stroke=0, fill=1)
    c.setStrokeColor(LINE_GRAY)
    c.setLineWidth(0.5)
    c.rect(bar_x, y, bar_w, bar_h, stroke=1, fill=0)
    for t in (25, 50, 75):
        tx = bar_x + bar_w * t / 100
        c.setStrokeColor(GRID_GRAY)
        c.setLineWidth(0.4)
        c.line(tx, y - 3, tx, y + bar_h + 3)
    c.setFillColor(_pctl_fill(score))
    c.rect(bar_x, y, bar_w * score_num / 100, bar_h, stroke=0, fill=1)
    c.setStrokeColor(NATS_WHITE)
    c.setLineWidth(1)
    c.line(bar_x + bar_w * score_num / 100, y, bar_x + bar_w * score_num / 100, y + bar_h)

    box_w = 26
    box_x = max(bar_x, min(bar_x + bar_w - box_w, bar_x + bar_w * score_num / 100 - box_w / 2))
    c.setFillColor(NATS_NAVY if score_num <= 55 else NATS_RED)
    c.rect(box_x, y, box_w, bar_h, stroke=0, fill=1)
    _fit_text(
        c,
        "-" if score is None else str(score_num),
        box_x + box_w / 2,
        y + 4,
        box_w - 4,
        align="center",
        max_size=6.8,
        min_size=5,
        fill=NATS_WHITE,
    )
    _fit_text(c, value, x + w - 8, y + 4, value_w, align="right", variant="regular", max_size=8.0, min_size=5)
    c.setStrokeColor(LINE_GRAY)
    c.setLineWidth(0.4)
    c.line(x + 8, y - 7, x + w - 8, y - 7)


def _draw_option_2_section(c: canvas.Canvas, *, y_top: float, title: str, rows: list[dict[str, Any]]) -> float:
    x = 50
    w = PAGE_W - 100
    row_gap = 22
    h = 41 + len(rows) * row_gap + 8
    y = y_top - h
    c.setFillColor(NATS_WHITE)
    c.rect(x, y, w, h, stroke=0, fill=1)
    c.setStrokeColor(LINE_GRAY)
    c.setLineWidth(0.75)
    c.rect(x, y, w, h, stroke=1, fill=0)
    _draw_option_2_section_header(c, x, y_top, title, w)
    row_y = y_top - 52
    for row in rows:
        _draw_option_2_metric_row(
            c,
            x + 10,
            row_y,
            w - 20,
            label=str(row["label"]),
            score=row.get("percentile"),
            value=str(row.get("value", "-")),
        )
        row_y -= row_gap
    return y - 10


def render_scorecard_option_2(
    data: dict[str, Any], output_path: str | Path | None = None
) -> bytes:
    """Render the structure-preserving Nationals scorecard skin."""
    c, buffer = _make_canvas()
    _draw_option_2_page_background(c)
    y = _draw_option_2_header(c, data)
    y = _draw_option_2_summary_grid(c, y, data.get("summary_cards", []))
    for section in data.get("sections", []):
        rows = section.get("rows", [])
        needed_h = 41 + len(rows) * 22 + 8
        if y - needed_h < 34:
            c.showPage()
            _draw_option_2_page_background(c)
            y = PAGE_H - 42
        y = _draw_option_2_section(
            c,
            y_top=y,
            title=str(section.get("title", "")),
            rows=rows,
        )
    c.showPage()
    c.save()
    return _finish_pdf(buffer, output_path)
