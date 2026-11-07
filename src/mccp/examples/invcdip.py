"""INV-CDIP: real scanned invoices for an extraction check.

350 real invoices (UCSF Industry Documents Library, mostly 1980s-1990s scans)
with human-labelled fields, from Gao et al., "Field Extraction from Forms with
Unlabeled Data" (ACL Spa-NLP 2022), https://github.com/salesforce/inv-cdip.

License: the annotations are CC BY-NC 4.0 (non-commercial); the documents fall
under the UCSF library's copyright and fair-use terms. MCCP downloads both at
evaluation time and never redistributes them: use the results as internal
evidence, not in a commercial product or its marketing without checking terms.

These invoices have no PO records or vendor master, so they cannot test the
triage decision (that part is code, verified on the other sets). They test the
part a model does on real documents: reading invoice number, PO number and the
amount payable from OCR text of a real scan.

Labels: a field the annotators did not label is *not scored* (an unlabelled PO
may still be printed on the page), so each field is scored only on the
documents where it is labelled, and a label that the PII redaction altered in
the input text is dropped the same way, so the model is never blamed for a value
it could not see. Download: page 1 of `<id>.pdf` from the public AWS Open Data
mirror (`s3://ucsf-idl-dataset`, anonymous HTTPS), cached on disk.
"""

from __future__ import annotations

import json
import re
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

ANNOTATIONS_COMMIT = "80a7c2cfb6838f4a02d04257444b528837d6fda9"
ANNOTATIONS_URL = f"https://codeload.github.com/salesforce/inv-cdip/tar.gz/{ANNOTATIONS_COMMIT}"
DOC_URL = "https://ucsf-idl-dataset.s3.amazonaws.com/{path}/{id}/{id}.pdf"
LICENSE_NOTE = "INV-CDIP annotations CC BY-NC 4.0; documents under UCSF IDL copyright/fair-use terms; internal use"

REALDOC_FIELDS = ["invoice_number", "po_number", "total_amount"]  # canonical, scored
REALDOC_MODEL_FIELDS = ["invoice_number", "po_number", "total_amount_raw"]  # what the model writes
REALDOC_SYSTEM_PROMPT = (
    "You read the OCR text of a scanned vendor invoice and extract three facts. Reply with only a JSON "
    "object with keys invoice_number, po_number and total_amount_raw. Copy invoice_number and po_number "
    "exactly as printed (the buyer's purchase order number, not a PO Box); use null when the invoice does "
    "not show one. For total_amount_raw copy the total amount payable exactly as printed, as text, "
    "including any currency symbol, or null; do not reformat it."
)
REALDOC_RAW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": REALDOC_MODEL_FIELDS,
    "properties": {
        "invoice_number": {"type": ["string", "null"]},
        "po_number": {"type": ["string", "null"]},
        "total_amount_raw": {"type": ["string", "null"]},
    },
}
REALDOC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": REALDOC_FIELDS,
    "properties": {
        "invoice_number": {"type": ["string", "null"]},
        "po_number": {"type": ["string", "null"]},
        "total_amount": {"type": ["number", "null"]},
    },
}
_LABELS = {"Invoice_number": "invoice_number", "Purchase_order": "po_number"}


def parse_amount(text: str) -> float | None:
    """'$1,533.54' -> 1533.54, '62000.00' -> 62000.0, '$3,000.' -> 3000.0.
    Anything ambiguous ('1,66667', '.00') -> None (that label is not scored).
    The one amount parser is mccp.rules.extraction.parse_money; this is its name here."""
    from mccp.rules.extraction import parse_money

    return parse_money(text)


def labels(annotation: dict[str, Any]) -> dict[str, Any]:
    """Expected values in the model's format; None = not labelled (not scored).
    total_amount_raw is the annotated text of the 'total amount' label, else of
    'amount due'; scoring parses it with the same code as the model's answer, and
    an annotation the parser cannot read ('1,66667') is simply not scored."""
    found: dict[str, str] = {}
    for f in annotation["Fields"]:
        found.setdefault(f["value"]["label"], f["value"]["tag"])
    out: dict[str, Any] = {name: found.get(label) for label, name in _LABELS.items()}
    raw_total = found.get("total_amount") or found.get("amount_due")
    out["total_amount_raw"] = raw_total if raw_total and parse_amount(raw_total) is not None else None
    return out


def doc_path(doc_id: str) -> str:
    return "/".join(doc_id[:4])


def fetch_annotations(cache: Path, opener: Callable[[str], bytes] | None = None) -> dict[str, dict[str, Any]]:
    """{doc_id: annotation} for the 350 labelled documents, from the pinned commit."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inv-cdip-{ANNOTATIONS_COMMIT[:12]}.tar.gz"
    if not path.exists():
        data = (opener or _http_get)(ANNOTATIONS_URL)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    out: dict[str, dict[str, Any]] = {}
    with tarfile.open(path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and "/annotation/" in m.name and m.name.endswith(".json"):
                f = tar.extractfile(m)
                if f is not None:
                    out[Path(m.name).stem] = json.load(f)
    if len(out) != 350:
        raise ValueError(f"expected 350 INV-CDIP annotations, found {len(out)} in {path}")
    return out


def fetch_pdf(doc_id: str, cache: Path, opener: Callable[[str], bytes] | None = None) -> bytes:
    if not re.fullmatch(r"[a-z]{4}\d{4}", doc_id):
        raise ValueError(f"unexpected INV-CDIP document id {doc_id!r}")
    path = cache / "pdf" / f"{doc_id}.pdf"
    if path.exists():
        return path.read_bytes()
    data = (opener or _http_get)(DOC_URL.format(path=doc_path(doc_id), id=doc_id))
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{doc_id}: download is not a PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mccp-eval/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https hosts above
        return r.read()


def first_page_image(pdf: bytes, dpi: int = 200):
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(stream=pdf, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


_ANNOTATION_LABELS = {
    "invoice_number": ("Invoice_number",),
    "po_number": ("Purchase_order",),
    "total_amount": ("total_amount", "amount_due"),  # the same order labels() uses
}


def label_box(annotation: dict[str, Any], field: str) -> tuple[float, float, float, float] | None:
    """Where the annotator saw the label, as fractions of the page (x0, y0, x1, y1),
    or None. Annotation boxes are in pixels of the original scan, whose size is
    `image_dims` = "[height, width, channels]"."""
    try:
        h, w = (float(v) for v in json.loads(annotation["image_dims"])[:2])
    except (KeyError, ValueError, TypeError):
        return None
    for want in _ANNOTATION_LABELS.get(field, ()):
        for f in annotation.get("Fields", []):
            b = (f.get("value") or {}).get("bbox")
            if (f.get("value") or {}).get("label") == want and b:
                return (b["xmin"] / w, b["ymin"] / h, b["xmax"] / w, b["ymax"] / h)
    return None


def audit_images(pdf: bytes, box: tuple[float, float, float, float] | None, dpi: int = 110):
    """(page, crop) PIL images of page 1 for a reviewer: the annotated label boxed in
    red on the page, and a crop around it (None without a box)."""
    from PIL import ImageDraw

    page = first_page_image(pdf, dpi=dpi).convert("RGB")
    if box is None:
        return page, None
    W, H = page.size
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    ImageDraw.Draw(page).rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), outline=(220, 0, 0), width=3)
    pad_x, pad_y = max(60.0, (x1 - x0)), max(40.0, 2 * (y1 - y0))
    crop = page.crop((max(0, x0 - pad_x), max(0, y0 - pad_y), min(W, x1 + pad_x), min(H, y1 + pad_y)))
    return page, crop


def to_realdoc_record(doc_id: str, text: str, expected: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    labelled = sorted(k for k, v in expected.items() if v is not None)
    return {
        "messages": [
            {"role": "system", "content": REALDOC_SYSTEM_PROMPT},
            {"role": "user", "content": f"OCR text of a scanned vendor invoice:\n\n{text}"},
            {"role": "assistant", "content": json.dumps(expected)},
        ],
        "meta": {
            "source_id": doc_id,
            "license": "cc-by-nc-4.0",
            "split": "invcdip",
            "scenario": "has_po" if expected.get("po_number") is not None else "no_po_label",
            "ood_kind": "+".join(labelled) or "none",
            **meta,
        },
    }


def generate_invcdip(
    cache: Path,
    n: int | None = None,
    opener: Callable[[str], bytes] | None = None,
    ocr: Callable[[Any], str] | None = None,
    log: Callable[[str], None] = lambda s: None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Records for the labelled documents (all 350, or the first `n` by id), plus a
    summary (downloaded, failed downloads, OCR cache hits). OCR text is cached per
    document, so a re-run costs no OCR."""
    from .documents import ocr_text, ocr_version

    ocr = ocr or ocr_text
    engine = ocr_version() if ocr is ocr_text else "custom"
    annotations = fetch_annotations(cache, opener)
    ids = sorted(annotations)[: n or None]
    out, failed, cached = [], [], 0
    for i, doc_id in enumerate(ids):
        exp = labels(annotations[doc_id])
        if all(v is None for v in exp.values()):
            continue
        txt_path = cache / "ocr" / f"{doc_id}.txt"
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8")
            cached += 1
        else:
            try:
                pdf = fetch_pdf(doc_id, cache, opener)
            except (OSError, ValueError) as e:
                failed.append({"id": doc_id, "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            text = ocr(first_page_image(pdf))
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(text, encoding="utf-8")
        if (i + 1) % 25 == 0:
            log(f"invcdip: {i + 1}/{len(ids)} documents")
        out.append(to_realdoc_record(doc_id, text, exp, {"ocr": engine}))
    summary = {
        "documents": len(out),
        "failed_downloads": failed,
        "ocr_cache_hits": cached,
        "license": LICENSE_NOTE,
        "annotations_commit": ANNOTATIONS_COMMIT,
    }
    return out, summary


def drop_redacted_labels(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    """`after` is `before` with PII redaction applied to the input text. A label
    that was in the OCR text before redaction and is gone after it becomes
    unscored (None): the model never saw it. A label the OCR itself lost stays
    scored, because OCR is part of the pipeline being measured. Returns how many
    labels were dropped; edits `after` in place."""
    dropped = 0
    for b, a in zip(before, after, strict=True):
        pre, post = _squash(b["messages"][1]["content"]), _squash(a["messages"][1]["content"])
        exp = json.loads(a["messages"][2]["content"])
        for k in ("invoice_number", "po_number"):
            v = exp.get(k)
            if v is not None and _squash(v) in pre and _squash(v) not in post:
                exp[k] = None
                dropped += 1
        a["messages"][2]["content"] = json.dumps(exp)
    return dropped


def ocr_ceiling(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Per id field: share of labelled values that appear (ignoring case, spaces and
    punctuation) in the OCR text. A model cannot beat this on that field, so a gap
    between it and the model's accuracy is the model's; the rest is OCR's."""
    out: dict[str, float | None] = {}
    for k in ("invoice_number", "po_number"):
        hits = [
            _squash(json.loads(r["messages"][2]["content"])[k]) in _squash(r["messages"][1]["content"])
            for r in records
            if json.loads(r["messages"][2]["content"]).get(k) is not None
        ]
        out[k] = round(sum(hits) / len(hits), 4) if hits else None
    return out


def _squash(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.casefold())


def normalized_equal(expected: Any, got: Any) -> bool:
    """Real-document field match: ids ignore case, spaces and punctuation
    ('#808448' == '808448'); amounts within half a cent; None only equals None."""
    if expected is None or got is None:
        return expected is None and got is None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(float(got) - expected) < 0.005
    return isinstance(got, str) and _squash(got) == _squash(str(expected)) and _squash(got) != ""
