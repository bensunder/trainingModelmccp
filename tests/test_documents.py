"""Realistic documents (render -> scan -> OCR) and real scans (INV-CDIP).

Runs the real renderer and the real OCR engine on a few pages (needs the `docs`
extra); network fetches are replaced by in-memory fakes."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import subprocess
import sys
import tarfile

import pytest
import yaml

pytest.importorskip("rapidocr_onnxruntime")
pytest.importorskip("reportlab")
pytest.importorskip("pymupdf")

from mccp.evals.benchmark import Target, run_benchmark  # noqa: E402
from mccp.evals.suite import EvalSuite  # noqa: E402
from mccp.examples import invcdip  # noqa: E402
from mccp.examples.documents import DOC_KINDS, generate_documents  # noqa: E402
from tests.conftest import ROOT  # noqa: E402

REALDOC = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-realdoc.yaml").read_text()))
EXTRACT = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-extract.yaml").read_text()))


# ------------------------------------------------------------------ documents --
@pytest.fixture(scope="module")
def docs():
    return generate_documents(4, seed=11)


def test_documents_are_extract_records_with_ocr_text_and_exact_labels(docs):
    assert [d["meta"]["ood_kind"] for d in docs] == list(DOC_KINDS[:4])
    for d in docs:
        assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant"]
        assert d["messages"][1]["content"].startswith("OCR text of a scanned vendor invoice:")
        assert d["meta"]["ocr"].startswith("rapidocr-onnxruntime") and d["meta"]["split"] == "docs"
        assert "lookups" in d["meta"] and "expected" in d["meta"]


def test_clean_scan_ocr_contains_the_facts(docs):
    clean = next(d for d in docs if d["meta"]["scan"] == "clean")
    text = clean["messages"][1]["content"].replace(" ", "")
    facts = json.loads(clean["messages"][2]["content"])
    for k in ("invoice_id", "po_number", "vendor_id"):
        assert facts[k].split("-")[1] in text, k
    assert f"{facts['unit_price_billed']:,.2f}" in text


def test_documents_are_deterministic():
    a, b = generate_documents(2, seed=3), generate_documents(2, seed=3)
    assert a == b


async def test_extract_suite_scores_documents_end_to_end(docs):
    gold = {d["messages"][1]["content"]: d["messages"][2]["content"] for d in docs}

    async def chat(payload):
        return {"choices": [{"message": {"content": gold[payload["messages"][-1]["content"]]}}], "usage": {}}

    rep = await run_benchmark(
        sets={"docs": docs}, targets=[Target("adapter", "adapter", "m", chat, {}, 4)], suite=EXTRACT
    )
    l1 = rep["sets"]["docs"]["targets"]["adapter"]["l1"]
    assert l1["decision_accuracy"]["mean"] == 1.0 and set(l1["by_ood_kind"]) == set(DOC_KINDS[:4])


def test_add_set_docs_cli_caches_the_generated_set(tmp_path):
    w = tmp_path / "w"
    (w / "data").mkdir(parents=True)
    (w / "pilot.json").write_text(json.dumps({"profile": "qwen3-8b@v1", "task": "extract"}))
    cmd = [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), "add-set", "--set", "docs", "--workdir", str(w)]
    cmd += ["--n", "2", "--cache", str(tmp_path / "cache")]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout.strip().splitlines()[-1])["cache"].startswith("written")
    second = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    summary = json.loads(second.stdout.strip().splitlines()[-1])
    assert summary["cache"] == "hit" and summary["records"] == 2
    rows = [json.loads(x) for x in (w / "data/golden_docs.jsonl").read_text().splitlines()]
    assert all(r["meta"]["task"] == "extract" for r in rows)
    assert (w / "set_docs.json").exists()


# ------------------------------------------------------------------- INV-CDIP --
@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("$1,533.54", 1533.54),
        ("62000.00", 62000.0),
        ("$3,000.", 3000.0),
        ("750", 750.0),
        ("$ 12,600.00", 12600.0),
        ("1,66667", None),
        (".00", None),
        ("approx 5", None),
    ],
)
def test_parse_amount(text, want):
    assert invcdip.parse_amount(text) == want


def _ann(**fields):
    return {"Fields": [{"key": {"tag": None}, "value": {"label": k, "tag": v}} for k, v in fields.items()]}


def test_labels_use_total_then_amount_due_and_leave_unlabelled_as_none():
    a = invcdip.labels(_ann(Invoice_number="9765-3009", total_amount="$755.00"))
    assert a == {"invoice_number": "9765-3009", "po_number": None, "total_amount": 755.0}
    b = invcdip.labels(_ann(Invoice_number="1", amount_due="$5,450.00", Purchase_order="M11794"))
    assert b == {"invoice_number": "1", "po_number": "M11794", "total_amount": 5450.0}


@pytest.mark.parametrize(
    ("exp", "got", "ok"),
    [
        ("#808448", "808448", True),
        ("9765-3009", "9765 3009", True),
        ("M10874-M-193", "m10874m193", True),
        ("9765-3009", "9765-3008", False),
        ("ABC", "", False),
        (755.0, 755.004, True),
        (755.0, 755.01, False),
        (755.0, "755.00", False),
        ("X1", None, False),
    ],
)
def test_normalized_equal(exp, got, ok):
    assert invcdip.normalized_equal(exp, got) is ok


def _fake_invcdip(n_docs: int = 350):
    """A tar.gz shaped like the GitHub archive, plus PDFs rendered with the facts."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    ids = [f"zz{chr(97 + i // 100)}{chr(97 + (i // 10) % 10)}{i:04d}" for i in range(n_docs)]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i, doc_id in enumerate(ids):
            fields = {"Invoice_number": f"77{i:04d}-A", "total_amount": f"${1000 + i:,.2f}"}
            if i % 3 == 0:
                fields["Purchase_order"] = f"PO{i:05d}"
            data = json.dumps(_ann(**fields)).encode()
            info = tarfile.TarInfo(f"inv-cdip-{invcdip.ANNOTATIONS_COMMIT}/annotation/{doc_id}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()

    def pdf_for(i: int) -> bytes:
        b = io.BytesIO()
        cv = canvas.Canvas(b, pagesize=letter, invariant=1)
        cv.setFont("Helvetica", 14)
        cv.drawString(72, 700, f"INVOICE NO  77{i:04d}-A")
        if i % 3 == 0:
            cv.drawString(72, 670, f"YOUR P.O. NO  PO{i:05d}")
        cv.drawString(72, 640, f"TOTAL  ${1000 + i:,.2f}")
        cv.showPage()
        cv.save()
        return b.getvalue()

    calls: list[str] = []

    def opener(url: str) -> bytes:
        calls.append(url)
        if url == invcdip.ANNOTATIONS_URL:
            return archive
        doc_id = url.rsplit("/", 1)[1][:-4]
        if doc_id == ids[1]:
            raise OSError("404")  # one missing document is reported, not fatal
        return pdf_for(ids.index(doc_id))

    return ids, opener, calls


def test_generate_invcdip_downloads_ocrs_caches_and_reports_failures(tmp_path):
    ids, opener, calls = _fake_invcdip()
    recs, info = invcdip.generate_invcdip(tmp_path, n=4, opener=opener)
    assert [r["meta"]["source_id"] for r in recs] == [ids[0], ids[2], ids[3]]
    assert len(info["failed_downloads"]) == 1 and info["failed_downloads"][0]["id"] == ids[1]
    assert all(r["meta"]["license"] == "cc-by-nc-4.0" for r in recs)
    assert f"{invcdip.doc_path(ids[0])}/{ids[0]}/{ids[0]}.pdf" in calls[1]
    exp = json.loads(recs[0]["messages"][2]["content"])
    assert exp == {"invoice_number": "770000-A", "po_number": "PO00000", "total_amount": 1000.0}
    assert invcdip.ocr_ceiling(recs)["invoice_number"] == 1.0
    n_calls = len(calls)
    again, info2 = invcdip.generate_invcdip(tmp_path, n=4, opener=opener)
    assert info2["ocr_cache_hits"] == 3 and len(calls) == n_calls + 1  # only the failed one retried
    assert [r["messages"] for r in again] == [r["messages"] for r in recs]


def test_redaction_drops_only_labels_it_removed_not_ocr_misses():
    def rec(text, inv):
        exp = {"invoice_number": inv, "po_number": None, "total_amount": None}
        return invcdip.to_realdoc_record("abcd0001", text, exp, {})

    before = [rec("INVOICE 04-90-0205-9251-560", "04-90-0205-9251-560"), rec("INVOICE 1234", "9999")]
    after = json.loads(json.dumps(before))
    after[0]["messages"][1]["content"] = "INVOICE <CREDIT_CARD_1>"
    assert invcdip.drop_redacted_labels(before, after) == 1
    assert json.loads(after[0]["messages"][2]["content"])["invoice_number"] is None
    assert json.loads(after[1]["messages"][2]["content"])["invoice_number"] == "9999"  # OCR miss stays scored


async def test_realdoc_scoring_skips_unlabelled_fields_and_reports_no_decision():
    exp = {"invoice_number": "#808448", "po_number": None, "total_amount": 1533.54}
    r = invcdip.to_realdoc_record("abcd0001", "text", exp, {})
    answers = iter(
        [
            {"invoice_number": "808448", "po_number": "PO Box 12", "total_amount": 1533.54},  # PO unlabelled: ignored
            {"invoice_number": "808449", "po_number": None, "total_amount": 1533.54},
        ]
    )

    async def chat(payload):
        return {"choices": [{"message": {"content": json.dumps(next(answers))}}], "usage": {}}

    async def score():
        t = Target("adapter", "adapter", "m", chat, {}, 1)
        rep = await run_benchmark(sets={"invcdip": [r]}, targets=[t], suite=REALDOC)
        return rep["sets"]["invcdip"]["targets"]["adapter"]["l1"]

    first, second = await score(), await score()
    assert first["exact_record_rate"] == 1.0 and first["decision_accuracy"] is None
    assert first["field_accuracy_by_field"] == {"invoice_number": 1.0, "total_amount": 1.0}
    assert second["field_accuracy_by_field"]["invoice_number"] == 0.0 and second["exact_record_rate"] == 0.0


def test_realdoc_suite_matches_the_code_contract():
    assert REALDOC.output_schema == invcdip.REALDOC_SCHEMA
    assert REALDOC.compare_fields == invcdip.REALDOC_FIELDS
    assert REALDOC.field_match == "normalized" and REALDOC.decision_rules is None


def test_evaluate_refuses_to_mix_invcdip_with_other_sets(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("gpu_pilot_docs", ROOT / "deploy/gpu/pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules["gpu_pilot_docs"] = pilot
    spec.loader.exec_module(pilot)
    w = tmp_path / "w"
    (w / "data").mkdir(parents=True)
    (w / "pilot.json").write_text(json.dumps({"profile": "qwen3-8b@v1", "task": "extract"}))
    for s in ("messy", "invcdip"):
        (w / "data" / pilot.SET_FILES[s]).write_text("{}\n")
    args = pilot.main.__globals__["argparse"].Namespace(
        workdir=str(w),
        targets="base",
        sets="messy,invcdip",
        report_name="r.json",
        adapter_name=None,
        vllm_url="http://127.0.0.1:9",
        vllm_api_key="k",
        timeout=1.0,
        max_items=None,
        concurrency=1,
        gpu_usd_per_hour=0.35,
        baseline_url=None,
        save_outputs=False,
    )
    assert asyncio.run(pilot._evaluate(args)) == 2
    assert "score invcdip on its own" in capsys.readouterr().err
