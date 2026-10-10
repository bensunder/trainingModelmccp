"""Adapter artifacts: content addressing and OpenSSF model signing.

Only the files vLLM needs are published (`adapter_config.json`,
`adapter_model.safetensors`). The artifact digest is the SHA-256 of a
canonical manifest of those files, so the storage path
`adapters/sha256/<digest>/` *is* the identity of the adapter.

Signing uses OpenSSF model-signing (Sigstore bundle with an in-toto
statement of per-file hashes). `key` mode signs with an ECDSA P-256 key (on
prem / air-gapped); `sigstore` mode signs keylessly with the workload's OIDC
identity in production. Verification checks the signature *and* that the
attested per-file digests equal our manifest, binding the two together.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mccp.domain.hashing import canonical_json, file_sha256, sha256_hex
from mccp.storage.objectstore import ObjectStore

PUBLISHED_FILES = ("adapter_config.json", "adapter_model.safetensors")
SIGNATURE_FILE = "model.sig"
MANIFEST_FILE = "manifest.json"


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterArtifact:
    digest: str  # hex sha256 of canonical manifest
    manifest: dict[str, Any]
    key_prefix: str  # object-store key prefix (no trailing slash)
    signature_sha256: str
    attested: dict[str, str]  # file -> sha256 as attested in the signature


def build_manifest(adapter_dir: Path) -> dict[str, Any]:
    files = []
    for name in PUBLISHED_FILES:
        p = adapter_dir / name
        if not p.is_file():
            raise ArtifactError(f"adapter is missing {name}")
        files.append({"path": name, "sha256": file_sha256(str(p)), "bytes": p.stat().st_size})
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ArtifactError("only PEFT LoRA adapters are supported")
    return {"format": "peft-lora", "files": files}


def manifest_digest(manifest: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(manifest))


class Signer:
    def __init__(self, mode: str, private_key: Path | None, public_key: Path | None) -> None:
        # Least privilege: only the host that registers adapters (GPU worker)
        # holds the private key; the control plane is verify-only (public key)
        # and the API needs neither. Missing keys fail at use, not at startup.
        self.mode = mode
        self.private_key = private_key
        self.public_key = public_key

    def sign(self, model_dir: Path, signature: Path) -> None:
        import model_signing

        cfg = model_signing.signing.Config()
        if self.mode == "key" and (self.private_key is None or not Path(self.private_key).is_file()):
            raise ArtifactError("signing needs MCCP_SIGNING_PRIVATE_KEY_PATH on this host (GPU worker only)")
        if self.mode == "key":
            cfg = cfg.use_elliptic_key_signer(private_key=str(self.private_key))
        else:
            cfg = cfg.use_sigstore_signer()
        cfg.sign(str(model_dir), str(signature))

    def verify(
        self, model_dir: Path, signature: Path, *, identity: str | None = None, oidc_issuer: str | None = None
    ) -> None:
        import model_signing

        cfg = model_signing.verifying.Config()
        if self.mode == "key" and (self.public_key is None or not Path(self.public_key).is_file()):
            raise ArtifactError("verification needs MCCP_SIGNING_PUBLIC_KEY_PATH on this host")
        if self.mode == "key":
            cfg = cfg.use_elliptic_key_verifier(public_key=str(self.public_key))
        else:
            if not identity or not oidc_issuer:
                raise ArtifactError("sigstore verification needs an expected identity and OIDC issuer")
            cfg = cfg.use_sigstore_verifier(identity=identity, oidc_issuer=oidc_issuer)
        try:
            cfg.verify(str(model_dir), str(signature))
        except Exception as e:  # model_signing raises ValueError on mismatch
            raise ArtifactError(f"signature verification failed: {e}") from e


def attested_digests(signature: Path) -> dict[str, str]:
    bundle = json.loads(signature.read_text(encoding="utf-8"))
    payload = json.loads(base64.b64decode(bundle["dsseEnvelope"]["payload"]))
    return {r["name"]: r["digest"] for r in payload["predicate"]["resources"]}


def _check_attestation(manifest: dict[str, Any], attested: dict[str, str]) -> None:
    want = {f["path"]: f["sha256"] for f in manifest["files"]}
    if want != attested:
        raise ArtifactError(f"signed file set/digests do not match manifest: manifest={want} attested={attested}")


async def publish_adapter(
    store: ObjectStore, tenant_prefix: str, adapter_dir: Path, staging: Path, signer: Signer
) -> AdapterArtifact:
    """Stage exactly the published files, sign them, upload content-addressed."""
    manifest = await asyncio.to_thread(build_manifest, adapter_dir)
    digest = manifest_digest(manifest)
    stage = staging / digest
    if stage.exists():
        shutil.rmtree(stage)
    model_dir = stage / "model"
    model_dir.mkdir(parents=True)
    for name in PUBLISHED_FILES:
        await asyncio.to_thread(shutil.copyfile, adapter_dir / name, model_dir / name)
    sig = stage / SIGNATURE_FILE
    await asyncio.to_thread(signer.sign, model_dir, sig)
    attested = attested_digests(sig)
    _check_attestation(manifest, attested)

    prefix = f"{tenant_prefix}/adapters/sha256/{digest}"
    for name in PUBLISHED_FILES:
        await store.put_file(f"{prefix}/{name}", model_dir / name)
    await store.put_file(f"{prefix}/{SIGNATURE_FILE}", sig)
    await store.put_bytes(f"{prefix}/{MANIFEST_FILE}", canonical_json(manifest))
    return AdapterArtifact(digest, manifest, prefix, file_sha256(str(sig)), attested)


async def materialize_verified(
    store: ObjectStore, key_prefix: str, expected_digest: str, dest: Path, signer: Signer
) -> Path:
    """Download an adapter, verify signature + manifest + digest, then move it
    into place atomically. Nothing unverified ever appears at `dest`."""
    tmp = dest.with_name(f".tmp-{dest.name}")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "model").mkdir(parents=True)
    try:
        manifest = json.loads(await store.get_bytes(f"{key_prefix}/{MANIFEST_FILE}"))
        if manifest_digest(manifest) != expected_digest:
            raise ArtifactError("manifest digest does not match the registered artifact digest")
        for name in PUBLISHED_FILES:
            await store.download_file(f"{key_prefix}/{name}", tmp / "model" / name)
        await store.download_file(f"{key_prefix}/{SIGNATURE_FILE}", tmp / SIGNATURE_FILE)
        await asyncio.to_thread(signer.verify, tmp / "model", tmp / SIGNATURE_FILE)
        _check_attestation(manifest, attested_digests(tmp / SIGNATURE_FILE))
        for f in manifest["files"]:
            if file_sha256(str(tmp / "model" / f["path"])) != f["sha256"]:
                raise ArtifactError(f"{f['path']} digest mismatch after download")
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        (tmp / "model").rename(dest)
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def generate_signing_keypair(private_path: Path, public_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    private_path.chmod(0o600)
    public_path.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
