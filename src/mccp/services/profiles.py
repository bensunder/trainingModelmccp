"""Model-profile validation workflow (CPU only; no GPU, no database).

`validate_profile` runs trainer/profile_check.py in the trainer environment
(transformers + huggingface_hub) against the real tokenizer, then on success
records the evidence (`profiles/validations/<ref>.json`) and pins the resolved
revision and chat-template hash into the profile YAML, which is what makes the
profile `validated`. A failed check changes nothing and returns the report.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mccp.domain.profile import ModelProfile, ProfileRegistry
from mccp.examples.invoices import generate, generate_ood

FIXTURE_COUNT = 24


def fixtures() -> list[dict[str, Any]]:
    """Records with tools and tool calls in several phrasings (in-distribution
    and OOD) so the render check covers varied content, not one template."""
    return generate(FIXTURE_COUNT // 2, seed=101) + generate_ood(FIXTURE_COUNT // 2, seed=202)


def build_spec(profile: ModelProfile) -> dict[str, Any]:
    return {
        "profile": profile.ref,
        "hf_id": profile.base.hf_id,
        "revision": profile.base.revision,
        "chat": profile.trainer_spec(),
        "dims": profile.dims.model_dump(),
        "fixtures": fixtures(),
    }


def _set_base_key(text: str, key: str, value: str) -> str:
    """Set `base.<key>` in a profile YAML, preserving comments and layout."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "base:")
    except StopIteration as e:
        raise ValueError("profile YAML has no top-level `base:` block") from e
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith(" ")), len(lines))
    pattern = re.compile(rf"^  {re.escape(key)}:")
    for i in range(start + 1, end):
        if pattern.match(lines[i]):
            lines[i] = f"  {key}: {value}"
            return "\n".join(lines) + "\n"
    after = next((i for i in range(start + 1, end) if lines[i].startswith("  hf_id:")), start)
    lines.insert(after + 1, f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def validate_profile(
    profiles_dir: Path, ref: str, *, trainer_python: str, check_script: Path, timeout_s: int = 900
) -> dict[str, Any]:
    registry = ProfileRegistry.load(profiles_dir)
    profile = registry.get(ref)
    with tempfile.TemporaryDirectory(prefix="mccp-validate-") as tmp:
        spec_path, out_path = Path(tmp) / "spec.json", Path(tmp) / "report.json"
        spec_path.write_text(json.dumps(build_spec(profile)), encoding="utf-8")
        proc = subprocess.run(
            [trainer_python, str(check_script), "--spec", str(spec_path), "--out", str(out_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if not out_path.exists():
            return {
                "profile": ref,
                "ok": False,
                "problems": [f"profile check did not run (exit {proc.returncode}): {proc.stderr[-800:]}"],
            }
        report = json.loads(out_path.read_text(encoding="utf-8"))
    if not report.get("ok"):
        return report

    # Pin, then record evidence for exactly what was checked.
    path = profiles_dir / f"{profile.ref}.yaml"
    text = path.read_text(encoding="utf-8")
    text = _set_base_key(text, "revision", report["revision"])
    text = _set_base_key(text, "chat_template_sha256", report["chat_template_sha256"])
    path.write_text(text, encoding="utf-8")
    record = {
        "validated_at": report["checked_at"],
        "method": "mccp profile validate (trainer/profile_check.py, real tokenizer, CPU)",
        "revision": report["revision"],
        "chat_template_sha256": report["chat_template_sha256"],
        "rendering_sha256": profile.rendering_sha256,
        "tool_render_checks": report["tool_render_checks"],
        "string_argument_checks": report["string_argument_checks"],
        "evidence": f"config architectures: {report.get('config_architectures')}",
    }
    vdir = profiles_dir / "validations"
    vdir.mkdir(exist_ok=True)
    (vdir / f"{profile.ref}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report["status"] = ProfileRegistry.load(profiles_dir).get(ref).status
    return report
