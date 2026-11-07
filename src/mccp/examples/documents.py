"""Realistic invoice documents for evaluation: render, scan, OCR.

Every other test set hands the model clean text. Real AP intake is a PDF or a
scan that goes through OCR first, so the model reads OCR output: lines in
reading order, layout columns merged, characters damaged by the scan. This set
reproduces that path end to end:

  case (labels from the policy) -> PDF in one of four layouts -> raster
  -> scan degradation (clean / light / heavy) -> OCR (RapidOCR, CPU) -> text

The facts and labels are exact by construction (the same `make_case` as every
other set, ids in their own range, 3,000,000+), so a miss is a reading error,
never a labelling error. The layouts carry what real invoices carry and the
synthetic text never did: a remit-to `PO Box` next to the PO number, bank
account digits, subtotal / tax / total beside the unit price, a freight line,
payment terms, dates and phone numbers.

Optional dependencies (`uv sync --extra docs`): reportlab, pymupdf, pillow,
rapidocr-onnxruntime. Imported lazily, so the rest of MCCP never needs them.
"""

from __future__ import annotations

import io
import random
from functools import lru_cache
from typing import Any

from .invoices import Case, make_case, record_builder
from .messy import _po_text, _vid_text

DOC_ID_OFFSET = 3_000_000
TEMPLATES = ("classic", "modern", "compact", "freight")
SCAN_LEVELS = ("clean", "light", "heavy")
DOC_KINDS = tuple(f"{t}:{s}" for t in TEMPLATES for s in SCAN_LEVELS)

_BUYER = ("Halcyon Energy Services", "4100 Industrial Pkwy", "Midland, TX 79706")
_STREETS = ["Commerce Dr", "Harbor Way", "Foundry Rd", "Riverside Ave", "Mill St", "Gateway Blvd"]
_CITIES = ["Houston, TX 77002", "Tulsa, OK 74103", "Denver, CO 80202", "Calgary, AB T2P 1J9", "Pittsburgh, PA 15222"]
_TERMS = ["Net 30", "Net 45", "2% 10, Net 30", "Due on receipt", "Net 60"]


def _fmt(x: float) -> str:
    return f"${x:,.2f}"


def _content(c: Case, rng: random.Random) -> dict[str, Any]:
    """Everything printed on the invoice. Ids are written in the canonical form or
    a house format (as on real documents); the label text varies by vendor."""
    house = rng.random() < 0.4
    tax_rate = rng.choice([0.0, 0.0, 0.0625, 0.0825, 0.05])
    amount = round(c.billed_qty * c.unit_price_billed, 2)
    freight = round(rng.uniform(25, 400), 2) if rng.random() < 0.5 else 0.0
    tax = round(amount * tax_rate, 2)
    return {
        "vendor": c.vendor_name,
        "vendor_addr": (
            f"{'PO Box ' + str(rng.randrange(100, 9999)) if rng.random() < 0.5 else str(rng.randrange(10, 9999)) + ' ' + rng.choice(_STREETS)}",
            rng.choice(_CITIES),
        ),
        "phone": f"({rng.randrange(201, 989)}) {rng.randrange(200, 999)}-{rng.randrange(1000, 9999)}",
        "email": c.contact_email,
        "invoice_label": rng.choice(["Invoice No.", "Invoice #", "Invoice Number", "INVOICE NO"]),
        "invoice": c.invoice_id,
        "po_label": rng.choice(["Customer PO", "Your Order No.", "P.O. Number", "Purchase Order", "Your Ref"]),
        "po": _po_text(c, rng, house),
        "vendor_label": rng.choice(["Vendor No.", "Supplier ID", "Supplier No.", "Vendor ID"]),
        "vendor_id": _vid_text(c, rng, house),
        "date": f"{rng.randrange(1, 13):02d}/{rng.randrange(1, 29):02d}/2026",
        "terms": rng.choice(_TERMS),
        "item": c.item,
        "sku": f"{rng.choice('ABCDEFGHJK')}{rng.randrange(1000, 99999)}",
        "qty": c.billed_qty,
        "price": c.unit_price_billed,
        "amount": amount,
        "freight": freight,
        "tax_rate": tax_rate,
        "tax": tax,
        "total": round(amount + freight + tax, 2),
        "bank": f"Remit to account {rng.randrange(10**9, 10**10)}, routing {rng.randrange(10**8, 10**9)}",
    }


def render_pdf(d: dict[str, Any], template: str) -> bytes:
    """One-page US-letter invoice in one of four layouts."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    cv = canvas.Canvas(buf, pagesize=letter, invariant=1)  # invariant: byte-stable output
    w, h = letter

    def text(x: float, y: float, s: str, size: int = 10, bold: bool = False) -> None:
        cv.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        cv.drawString(x, y, s)

    def rtext(x: float, y: float, s: str, size: int = 10, bold: bool = False) -> None:
        cv.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        cv.drawRightString(x, y, s)

    lines = [(d["sku"], d["item"], d["qty"], d["price"], d["amount"])]
    if template == "freight" or d["freight"]:
        lines.append(("FRT", "Freight and handling", 1, d["freight"] or 85.0, d["freight"] or 85.0))
    subtotal = round(sum(x[4] for x in lines), 2)
    total = round(subtotal + d["tax"], 2)

    if template == "classic":
        text(50, h - 60, d["vendor"], 16, True)
        text(50, h - 78, d["vendor_addr"][0])
        text(50, h - 92, d["vendor_addr"][1])
        text(50, h - 106, f"Tel {d['phone']}  {d['email']}", 9)
        rtext(w - 50, h - 60, "INVOICE", 20, True)
        for i, (k, v) in enumerate(
            [
                (d["invoice_label"], d["invoice"]),
                ("Date", d["date"]),
                (d["po_label"], d["po"]),
                (d["vendor_label"], d["vendor_id"]),
                ("Terms", d["terms"]),
            ]
        ):
            rtext(w - 170, h - 90 - 15 * i, f"{k}:", 10, True)
            text(w - 165, h - 90 - 15 * i, str(v))
        text(50, h - 190, "Bill To:", 10, True)
        for i, s in enumerate(_BUYER):
            text(50, h - 205 - 13 * i, s)
        y0 = h - 280
    elif template == "modern":
        cv.setFillGray(0.85)
        cv.rect(0, h - 90, w, 90, fill=1, stroke=0)
        cv.setFillGray(0)
        text(40, h - 50, d["vendor"].upper(), 18, True)
        text(40, h - 70, f"{d['vendor_addr'][0]}, {d['vendor_addr'][1]}  |  {d['phone']}", 9)
        text(40, h - 120, "Invoice", 22, True)
        grid = [
            (d["invoice_label"], d["invoice"]),
            (d["vendor_label"], d["vendor_id"]),
            (d["po_label"], d["po"]),
            ("Invoice Date", d["date"]),
        ]
        for i, (k, v) in enumerate(grid):
            x = 40 + (i % 2) * 270
            y = h - 150 - (i // 2) * 30
            text(x, y, k, 8)
            text(x, y - 12, str(v), 11, True)
        text(40, h - 225, "Billed to  " + ", ".join(_BUYER), 9)
        y0 = h - 270
    elif template == "compact":
        text(50, h - 50, f"{d['vendor']} - {d['vendor_addr'][0]} - {d['vendor_addr'][1]}", 11, True)
        text(50, h - 66, f"{d['invoice_label']} {d['invoice']}    Date {d['date']}    {d['terms']}", 10)
        text(50, h - 80, f"{d['po_label']}: {d['po']}    {d['vendor_label']}: {d['vendor_id']}", 10)
        text(50, h - 94, f"Ship to: {_BUYER[0]}, {_BUYER[2]}", 9)
        y0 = h - 130
    else:  # freight: remittance-style header, line table with a freight line
        text(50, h - 55, "INVOICE / REMITTANCE", 14, True)
        text(50, h - 75, d["vendor"], 12, True)
        text(50, h - 89, d["vendor_addr"][0] + ", " + d["vendor_addr"][1])
        text(350, h - 75, f"{d['invoice_label']}: {d['invoice']}")
        text(350, h - 89, f"{d['po_label']}: {d['po']}")
        text(350, h - 103, f"{d['vendor_label']}: {d['vendor_id']}")
        text(350, h - 117, f"Date: {d['date']}   Terms: {d['terms']}")
        y0 = h - 160

    cols = [50, 120, 360, 430, 540]
    for x, head in zip(cols, ["Item", "Description", "Qty", "Unit Price", "Amount"], strict=True):
        (rtext if x >= 360 else text)(x if x < 360 else x + 40, y0, head, 10, True)
    cv.line(50, y0 - 4, w - 40, y0 - 4)
    for i, (sku, desc, qty, price, amt) in enumerate(lines):
        y = y0 - 20 - 16 * i
        text(50, y, sku)
        text(120, y, desc)
        rtext(400, y, str(qty))
        rtext(470, y, _fmt(price))
        rtext(580, y, _fmt(amt))
    y = y0 - 30 - 16 * len(lines)
    cv.line(360, y + 8, w - 40, y + 8)
    for k, v in (("Subtotal", subtotal), (f"Tax ({d['tax_rate'] * 100:.2f}%)", d["tax"]), ("TOTAL DUE", total)):
        rtext(470, y - 6, k, 10, k == "TOTAL DUE")
        rtext(580, y - 6, _fmt(v), 10, k == "TOTAL DUE")
        y -= 16
    text(50, 90, d["bank"], 8)
    text(50, 76, f"Questions: {d['email']}  {d['phone']}   Thank you for your business.", 8)
    cv.showPage()
    cv.save()
    return buf.getvalue()


def rasterize(pdf: bytes, dpi: int = 150):
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(stream=pdf, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def _degrade_params(level: str, rng: random.Random) -> tuple | None:
    """Draw the scan damage for one page. Always called, rendered or not, so the
    random stream (and every later case) is identical whether a page is rendered
    and OCR'd or its OCR text comes from the cache."""
    if level == "clean":
        return None
    heavy = level == "heavy"
    angle = rng.uniform(-2.0, 2.0) if heavy else rng.uniform(-0.7, 0.7)
    blur = rng.uniform(0.8, 1.3) if heavy else rng.uniform(0.3, 0.6)
    noise_seed = rng.randrange(2**32)
    return heavy, angle, blur, noise_seed


def _apply_degrade(img, params: tuple | None):
    import numpy as np
    from PIL import Image, ImageFilter

    if params is None:
        return img
    heavy, angle, blur, noise_seed = params
    img = img.rotate(angle, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, dtype=np.int16)
    noise = np.random.default_rng(noise_seed).normal(0, 22 if heavy else 9, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if heavy:
        w, h = img.size
        img = img.resize((int(w * 0.7), int(h * 0.7))).resize((w, h))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=35 if heavy else 70)
    buf.seek(0)
    return Image.open(buf).convert("L")


def degrade(img, level: str, rng: random.Random):
    """Scanner damage: skew, blur, speckle, low-resolution JPEG. Seeded."""
    return _apply_degrade(img, _degrade_params(level, rng))


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_version() -> str:
    import importlib.metadata as md

    return f"rapidocr-onnxruntime {md.version('rapidocr-onnxruntime')} (no angle classifier)"


def ocr_text(img) -> str:
    """OCR one page into reading-order text: boxes grouped into lines by vertical
    overlap, each line left to right, wide gaps kept as double spaces. This is
    the plain text an OCR step hands an LLM in production."""
    import numpy as np

    # use_cls=False: pages are upright; the angle classifier flips short skewed
    # boxes 180 degrees ("90" read as "06"), an engine artifact, not scan damage.
    result, _ = _engine()(np.asarray(img.convert("RGB")), use_cls=False)
    if not result:
        return ""
    boxes = []
    for quad, txt, _score in result:
        ys = [p[1] for p in quad]
        xs = [p[0] for p in quad]
        boxes.append((min(ys), max(ys), min(xs), max(xs), txt))
    boxes.sort(key=lambda b: ((b[0] + b[1]) / 2, b[2]))
    lines: list[list[tuple]] = []
    for b in boxes:
        mid = (b[0] + b[1]) / 2
        if lines:
            top = min(x[0] for x in lines[-1])
            bot = max(x[1] for x in lines[-1])
            if top <= mid <= bot:
                lines[-1].append(b)
                continue
        lines.append([b])
    out = []
    for line in lines:
        line.sort(key=lambda b: b[2])
        parts, prev_right = [], None
        for b in line:
            if prev_right is not None:
                parts.append("  " if b[2] - prev_right > 40 else " ")
            parts.append(b[4])
            prev_right = b[3]
        out.append("".join(parts).strip())
    return "\n".join(x for x in out if x)


def generate_documents(
    n: int,
    seed: int = 5151,
    task: str = "extract",
    keep_images: list | None = None,
    ocr_texts: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """`n` invoice documents, kinds (layout x scan level) round-robin. The model's
    input is the OCR text; `keep_images`, when a list, receives (source_id, PNG
    bytes) for the first page of each document (audit samples). `ocr_texts`
    ({invoice_id: OCR text} from an earlier run of this generator, same seed)
    skips rendering and OCR for those pages: the cases, labels and prompts are
    rebuilt by the current code, only the expensive OCR is reused."""
    if n < 1:
        raise ValueError("n must be >= 1")
    build = record_builder(task)
    rng = random.Random(seed)
    engine = ocr_version()
    out = []
    for i in range(n):
        c = make_case(rng, i + DOC_ID_OFFSET)
        kind = DOC_KINDS[i % len(DOC_KINDS)]
        template, level = kind.split(":")
        d = _content(c, rng)
        params = _degrade_params(level, rng)
        cached = (ocr_texts or {}).get(c.invoice_id)
        if cached is not None and keep_images is None:
            text = cached
        else:
            img = _apply_degrade(rasterize(render_pdf(d, template)), params)
            text = ocr_text(img)
            if keep_images is not None:
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                keep_images.append((c.invoice_id, buf.getvalue()))
        meta = {"split": "docs", "ood_kind": kind, "doc_template": template, "scan": level, "ocr": engine}
        out.append(build(c, f"{OCR_PREFIX}{text}", meta))
    return out


OCR_PREFIX = "OCR text of a scanned vendor invoice:\n\n"
RENDER_VERSION = "1"  # bump when rendering, degradation or OCR settings change (invalidates OCR caches)
# documents.py file hashes (first 12 hex of sha256) whose rendering is RENDER_VERSION 1:
# their OCR caches (named docs-<task>-n<N>-s<seed>-<hash>.jsonl) hold the same text.
# Verified pixel-identical when degrade() was split into _degrade_params/_apply_degrade.
LEGACY_RENDER_HASHES = ("5857fcb9f107",)


def ocr_texts_from_records(records: list[dict[str, Any]]) -> dict[str, str]:
    """{invoice_id: OCR text} from records this generator produced (any task)."""
    out = {}
    for r in records:
        text = r["messages"][1]["content"]
        if text.startswith(OCR_PREFIX):
            out[r["meta"]["source_id"]] = text[len(OCR_PREFIX) :]
    return out
