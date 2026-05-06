"""Generate the submission PDF: cover page in the required format + condensed report.

Run after `python -m src.main --all` so figures exist in `report/figures/`.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
METRICS_PATH = REPORT_DIR / "metrics.json"

SUBMISSION = {
    "exercise_number": "1",
    "group_id": "SMNGRP05",
    "self_score": "90",
    "students": [
        {
            "id": "208123232",
            "first_en": "Afaf",
            "last_en": "Gharra",
            "first_he": "עפאף",
            "last_he": "גרה",
        },
        {
            "id": "212018899",
            "first_en": "Reem",
            "last_en": "Awawdy",
            "first_he": "רים",
            "last_he": "עואודה",
        },
    ],
    "github": "https://github.com/afaf-gharra/SignalMemoryNet",
    "late_submission": "no",
}


def _register_unicode_font() -> str:
    """Register a TTF that supports Hebrew. Returns the font name to use.

    Falls back to ``Helvetica`` if no suitable system font is found, in which
    case Hebrew characters will likely render as boxes.
    """
    candidates = [
        ("HebrewMain", r"C:\Windows\Fonts\arial.ttf"),
        ("HebrewMain", r"C:\Windows\Fonts\ARIALUNI.TTF"),
        ("HebrewMain", "/Library/Fonts/Arial.ttf"),
        ("HebrewMain", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for name, path in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return "Helvetica"


def _styles():
    base = getSampleStyleSheet()
    body_font = _register_unicode_font()
    base.add(ParagraphStyle(name="H1Big", fontName=body_font,
                            fontSize=18, spaceAfter=14))
    base.add(ParagraphStyle(name="Field", fontName=body_font,
                            fontSize=12, leading=18))
    base.add(ParagraphStyle(name="FieldBold", fontName=body_font,
                            fontSize=12, leading=18, spaceBefore=8))
    base.add(ParagraphStyle(name="Body", fontName=body_font,
                            fontSize=11, leading=15, spaceAfter=8))
    return base


def _cover(story, styles) -> None:
    story.append(Paragraph("Submission cover sheet", styles["H1Big"]))

    story.append(Paragraph(
        f"<b>1. Submitting an exercise number:</b> {SUBMISSION['exercise_number']}",
        styles["Field"]))
    story.append(Paragraph(
        f"<b>2. Group ID code (8 characters in English without spaces):</b> "
        f"{SUBMISSION['group_id']}",
        styles["Field"]))
    story.append(Paragraph(
        f"<b>3. Recommendation for self-scoring for the group on submission:</b> "
        f"{SUBMISSION['self_score']}",
        styles["Field"]))

    for i, s in enumerate(SUBMISSION["students"], start=1):
        story.append(Paragraph(f"<b>{3 + i}. Student {i}</b>", styles["FieldBold"]))
        story.append(Paragraph(f"ID card: {s['id']}", styles["Field"]))
        story.append(Paragraph(f"First name in English: {s['first_en']}", styles["Field"]))
        story.append(Paragraph(f"Last name in English: {s['last_en']}", styles["Field"]))
        story.append(Paragraph(f"First name in Hebrew: {s['first_he']}", styles["Field"]))
        story.append(Paragraph(f"Last name in Hebrew: {s['last_he']}", styles["Field"]))

    story.append(Paragraph(
        f"<b>6. Link to GITHUB:</b> "
        f"<font color='blue'>{SUBMISSION['github']}</font>",
        styles["Field"]))
    story.append(Paragraph(
        f"<b>7. A late submission confirmation (yes/no) is attached to this PDF "
        f"document on the next page:</b> {SUBMISSION['late_submission']}",
        styles["Field"]))
    story.append(PageBreak())


def _report_body(story, styles) -> None:
    story.append(Paragraph("SignalMemoryNet — HW1 Lab Report", styles["H1Big"]))

    story.append(Paragraph(
        "We trained three architectures (MLP, RNN, LSTM) on a synthetic "
        "multi-frequency sine task. Each input is a noisy 10-sample window of "
        "a sine wave with frequency in {2, 5, 10, 20} Hz sampled at 100 Hz. "
        "Each model jointly predicts the 1-hot frequency code (cross-entropy) "
        "and reconstructs the corresponding clean window (MSE). Loss = "
        "CE + MSE.",
        styles["Body"]))

    story.append(Paragraph(
        "Dataset: amplitude A ∈ [0.8, 1.2], phase φ uniform in [0, 2π), noise "
        "σ ∈ {5%, 10%, 20%} of A, additive Gaussian. Each record is generated "
        "twice (clean + noisy) with shared (A, φ). Sliding window with stride 5.",
        styles["Body"]))

    story.append(Paragraph(
        "Architectures: MLP backbone Linear(10,64)-ReLU-Linear(64,64)-ReLU. "
        "RNN backbone nn.RNN(in=1,hidden=32,tanh) over 10 timesteps. LSTM "
        "backbone nn.LSTM(in=1,hidden=32). All share a multi-task linear head "
        "producing 4 logits + 10 reconstruction outputs. Optimiser Adam, "
        "lr=1e-3, batch 128, early stopping on val loss.",
        styles["Body"]))

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        rows = [["Model", "Test acc", "Recon MSE"]]
        for name, m in metrics.items():
            rows.append([name.upper(), f"{m['acc']:.3f}", f"{m['mse']:.4f}"])
        tbl = Table(rows, hAlign="LEFT", colWidths=[3 * cm, 3 * cm, 3 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("<b>Test-set results</b>", styles["FieldBold"]))
        story.append(tbl)
    else:
        story.append(Paragraph(
            "<i>metrics.json not found — run `python -m src.main --all` first "
            "to populate test-set numbers.</i>",
            styles["Body"]))

    story.append(Spacer(1, 0.4 * cm))

    figures = [
        ("training_curves.png", "Training / validation curves for all three models."),
        ("confusion_mlp.png",   "Confusion matrix — MLP."),
        ("confusion_rnn.png",   "Confusion matrix — RNN."),
        ("confusion_lstm.png",  "Confusion matrix — LSTM."),
        ("recon_mlp.png",       "Reconstruction examples — MLP."),
        ("recon_rnn.png",       "Reconstruction examples — RNN."),
        ("recon_lstm.png",      "Reconstruction examples — LSTM."),
    ]
    for fname, caption in figures:
        path = FIG_DIR / fname
        if path.exists():
            story.append(Image(str(path), width=15 * cm, height=8 * cm,
                               kind="proportional"))
            story.append(Paragraph(f"<i>{caption}</i>", styles["Body"]))
            story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "Discussion: the vanilla RNN tends to do better than the MLP on the "
        "20 Hz signal because high-frequency content needs only short-term "
        "memory, which the RNN handles before vanishing-gradient effects "
        "matter (lecture). The LSTM, with its gated cell state, is more "
        "robust on lower frequencies (2–5 Hz) and at the highest noise level "
        "(σ = 20 %), which matches the theoretical advantage of element-wise "
        "gating for selectively preserving relevant information.",
        styles["Body"]))


def main(out: Path | None = None) -> Path:
    out = out or (REPORT_DIR / "Submission.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="SignalMemoryNet — HW1 Submission")
    styles = _styles()
    story: list = []
    _cover(story, styles)
    _report_body(story, styles)
    doc.build(story)
    print("Wrote", out)
    return out


if __name__ == "__main__":
    main()
