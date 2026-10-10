"""Object storage behind one small interface.

Two implementations: local filesystem (dev / single host / NFS volume) and
S3-compatible (AWS S3, MinIO, Azure via an S3 gateway). Keys are always
relative, slash-separated and validated, so callers cannot escape the root.
Content-addressed layout:

    tenants/<tenant>/datasets/<project>/<name>/v<N>/...
    tenants/<tenant>/adapters/sha256/<digest>/...
    tenants/<tenant>/evidence/<sha256>.json
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

_KEY = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.?(?:/|$))[A-Za-z0-9._\-/@:+=]{1,1024}$")


class StorageError(RuntimeError):
    pass


def check_key(key: str) -> str:
    if not _KEY.fullmatch(key) or "//" in key:
        raise StorageError(f"invalid object key: {key!r}")
    return key


class ObjectStore(Protocol):
    url: str

    async def put_bytes(self, key: str, data: bytes) -> str: ...
    async def put_file(self, key: str, path: Path) -> str: ...
    async def get_bytes(self, key: str) -> bytes: ...
    async def download_file(self, key: str, dest: Path) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def list(self, prefix: str) -> list[str]: ...
    async def ensure_bucket(self) -> None: ...
    def uri(self, key: str) -> str: ...


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.url = f"file://{self.root}"

    def _path(self, key: str) -> Path:
        p = (self.root / check_key(key)).resolve()
        if self.root not in p.parents:
            raise StorageError(f"key escapes store root: {key!r}")
        return p

    def uri(self, key: str) -> str:
        return f"{self.url}/{check_key(key)}"

    async def ensure_bucket(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write(dest: Path, writer) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                writer(f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    async def put_bytes(self, key: str, data: bytes) -> str:
        dest = self._path(key)
        await asyncio.to_thread(self._atomic_write, dest, lambda f: f.write(data))
        return self.uri(key)

    async def put_file(self, key: str, path: Path) -> str:
        dest = self._path(key)

        def copy(f):
            with open(path, "rb") as src:
                shutil.copyfileobj(src, f, length=1 << 20)

        await asyncio.to_thread(self._atomic_write, dest, copy)
        return self.uri(key)

    async def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        try:
            return await asyncio.to_thread(p.read_bytes)
        except FileNotFoundError as e:
            raise StorageError(f"object not found: {key}") from e

    async def download_file(self, key: str, dest: Path) -> None:
        src = self._path(key)
        if not src.is_file():
            raise StorageError(f"object not found: {key}")

        def copy(f):
            with open(src, "rb") as s:
                shutil.copyfileobj(s, f, length=1 << 20)

        await asyncio.to_thread(self._atomic_write, dest, copy)

    async def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    async def list(self, prefix: str) -> list[str]:
        base = self._path(prefix.rstrip("/")) if prefix else self.root

        def walk() -> list[str]:
            if not base.exists():
                return []
            return sorted(
                str(p.relative_to(self.root)).replace(os.sep, "/")
                for p in base.rglob("*")
                if p.is_file() and not p.name.startswith(".tmp-")
            )

        return await asyncio.to_thread(walk)


class S3Store:
    """S3-compatible store (AWS S3, MinIO, Ceph RGW). Credentials come from the
    standard AWS chain (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env, instance or
    workload identity). Path-style addressing so MinIO works without wildcard DNS."""

    def __init__(self, bucket: str, prefix: str, endpoint_url: str | None, region: str | None = None) -> None:
        import boto3
        from botocore.config import Config

        cfg = Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=10,
            read_timeout=120,
        )
        self._client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region or "us-east-1", config=cfg)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.url = f"s3://{bucket}/{self.prefix}".rstrip("/")

    def _k(self, key: str) -> str:
        check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    @staticmethod
    def _not_found(e: Exception) -> bool:
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        return code in ("404", "NoSuchKey", "NotFound")

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._k(key)}"

    async def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            await asyncio.to_thread(self._client.head_bucket, Bucket=self.bucket)
        except ClientError as e:
            if str(e.response.get("Error", {}).get("Code")) not in ("404", "NoSuchBucket", "NotFound"):
                raise
            await asyncio.to_thread(self._client.create_bucket, Bucket=self.bucket)

    async def put_bytes(self, key: str, data: bytes) -> str:
        await asyncio.to_thread(self._client.put_object, Bucket=self.bucket, Key=self._k(key), Body=data)
        return self.uri(key)

    async def put_file(self, key: str, path: Path) -> str:
        await asyncio.to_thread(self._client.upload_file, str(path), self.bucket, self._k(key))
        return self.uri(key)

    async def get_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            obj = await asyncio.to_thread(self._client.get_object, Bucket=self.bucket, Key=self._k(key))
            return await asyncio.to_thread(obj["Body"].read)
        except ClientError as e:
            if self._not_found(e):
                raise StorageError(f"object not found: {key}") from e
            raise

    async def download_file(self, key: str, dest: Path) -> None:
        from botocore.exceptions import ClientError

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".tmp-{dest.name}")
        try:
            await asyncio.to_thread(self._client.download_file, self.bucket, self._k(key), str(tmp))
            os.replace(tmp, dest)
        except ClientError as e:
            if self._not_found(e):
                raise StorageError(f"object not found: {key}") from e
            raise
        finally:
            tmp.unlink(missing_ok=True)

    async def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            await asyncio.to_thread(self._client.head_object, Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError as e:
            # Only "not found" means absent; auth/network errors must surface.
            if self._not_found(e):
                return False
            raise

    async def list(self, prefix: str) -> list[str]:
        def run() -> list[str]:
            out: list[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            # Directory semantics, same as LocalStore: "t/a" lists t/a/*, not t/ab.
            full = (self._k(prefix.rstrip("/")) + "/") if prefix else (self.prefix + "/" if self.prefix else "")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
                for item in page.get("Contents", []):
                    k = item["Key"]
                    out.append(k[len(self.prefix) + 1 :] if self.prefix else k)
            return sorted(out)

        return await asyncio.to_thread(run)


def open_store(url: str, s3_endpoint_url: str | None = None, s3_region: str | None = None) -> ObjectStore:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return LocalStore(Path(parsed.path))
    if parsed.scheme == "s3":
        return S3Store(parsed.netloc, parsed.path, s3_endpoint_url, s3_region)
    raise StorageError(f"unsupported object store url: {url}")


def tenant_prefix(tenant_id: object) -> str:
    return f"tenants/{tenant_id}"
