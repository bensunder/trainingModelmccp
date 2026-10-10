"""Contract test against the real Soup package.

soup-cli pins typer<0.21, which conflicts with the control plane, so it lives in
its own interpreter. Point MCCP_TEST_SOUP_PYTHON at a Python that has
`soup-cli==<recipe soup.version>` installed (no GPU/torch needed).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from mccp.datasets.pii import PatternRedactor
from mccp.datasets.pipeline import build_dataset
from mccp.domain.recipe import Recipe
from mccp.domain.spec import JobRequest
from mccp.examples.invoices import generate
from mccp.training.soup_config import build_soup_yaml
from tests.conftest import ROOT

SOUP_PY = os.environ.get("MCCP_TEST_SOUP_PYTHON")
pytestmark = pytest.mark.skipif(not SOUP_PY, reason="set MCCP_TEST_SOUP_PYTHON to a python with soup-cli installed")

CHECK = r"""
import json, sys
from soup_cli.config.loader import load_config_from_string
from soup_cli.data.formats import format_to_messages
cfg = load_config_from_string(open(sys.argv[1]).read())
rows = [json.loads(l) for l in open(sys.argv[2])]
ok = sum(format_to_messages(r, cfg.data.format) is not None for r in rows)
print(json.dumps({"task": cfg.task, "quant": cfg.training.quantization, "rows": len(rows), "ok": ok}))
"""


def test_generated_config_and_data_are_accepted_by_soup(tmp_path: Path):
    raw = yaml.safe_load((ROOT / "examples/recipe-qwen3-8b-qlora-sft.yaml").read_text())
    raw["base_model"]["revision"] = "9c925d64d72725edaf899c6cb9c377fd0709d9c5"
    cfg = JobRequest.model_validate(yaml.safe_load((ROOT / "examples/job-invoice-triage.yaml").read_text())).resolve(
        Recipe.model_validate(raw)
    )
    res = build_dataset(b"".join(json.dumps(r).encode() + b"\n" for r in generate(200)), PatternRedactor())
    (tmp_path / "train.jsonl").write_bytes(res.files["train.jsonl"])
    soup_yaml = build_soup_yaml(cfg, n_train=100, n_val=10)
    soup_yaml["base"] = "./model"
    (tmp_path / "soup.yaml").write_text(yaml.safe_dump(soup_yaml))
    out = subprocess.run(
        [SOUP_PY, "-c", CHECK, str(tmp_path / "soup.yaml"), str(tmp_path / "train.jsonl")],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["task"] == "sft" and result["quant"] == "4bit"
    assert result["ok"] == result["rows"] > 0
