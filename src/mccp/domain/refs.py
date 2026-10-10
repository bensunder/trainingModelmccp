"""Strictly parsed references. These strings become object-store prefixes and
lineage keys, so parsing rejects anything that is not a plain slug."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer

from .errors import InvalidReference

_SLUG = re.compile(r"^[a-z](?:[a-z0-9]|-(?=[a-z0-9])){0,62}$")


def is_slug(s: str) -> bool:
    return bool(_SLUG.fullmatch(s))


def require_slug(field: str, s: str) -> str:
    if not is_slug(s):
        raise InvalidReference(f"{field} `{s}` must be a lowercase slug (a-z, 0-9, single dashes, <= 63 chars)")
    return s


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """`project/name@vN[:split]`"""

    project: str
    name: str
    version: int
    split: Split | None = None

    _RE = re.compile(r"^(?P<project>[^/@:]+)/(?P<name>[^/@:]+)@v(?P<version>[1-9][0-9]{0,8})(?::(?P<split>[a-z]+))?$")

    @classmethod
    def parse(cls, s: str) -> "DatasetRef":
        m = cls._RE.fullmatch(s.strip()) if isinstance(s, str) else None
        if not m:
            raise InvalidReference(f"dataset ref `{s}` must look like `project/name@vN[:split]`")
        project, name = m["project"], m["name"]
        if not (is_slug(project) and is_slug(name)):
            raise InvalidReference(f"dataset ref `{s}`: project and name must be lowercase slugs")
        split = None
        if m["split"]:
            try:
                split = Split(m["split"])
            except ValueError as e:
                raise InvalidReference(f"dataset ref `{s}`: split must be train, val or test") from e
        return cls(project, name, int(m["version"]), split)

    def with_split(self, split: Split) -> "DatasetRef":
        return DatasetRef(self.project, self.name, self.version, split)

    @property
    def version_ref(self) -> str:
        return f"{self.project}/{self.name}@v{self.version}"

    def __str__(self) -> str:
        return self.version_ref + (f":{self.split.value}" if self.split else "")


@dataclass(frozen=True, slots=True)
class RecipeRef:
    """`name@vMAJOR.MINOR.PATCH+GPU`"""

    name: str
    version: tuple[int, int, int]
    gpu: str

    _RE = re.compile(
        r"^(?P<name>[^@+]+)@v?(?P<a>\d{1,6})\.(?P<b>\d{1,6})\.(?P<c>\d{1,6})\+(?P<gpu>[A-Za-z0-9-]{1,32})$"
    )

    @classmethod
    def parse(cls, s: str) -> "RecipeRef":
        m = cls._RE.fullmatch(s.strip()) if isinstance(s, str) else None
        if not m or not is_slug(m["name"]):
            raise InvalidReference(f"recipe ref `{s}` must look like `name@vX.Y.Z+GPU`")
        return cls(m["name"], (int(m["a"]), int(m["b"]), int(m["c"])), m["gpu"])

    def __str__(self) -> str:
        a, b, c = self.version
        return f"{self.name}@v{a}.{b}.{c}+{self.gpu}"


@dataclass(frozen=True, slots=True)
class SuiteRef:
    """`name@vN` for evaluation suites."""

    name: str
    version: int

    _RE = re.compile(r"^(?P<name>[^@]+)@v(?P<v>[1-9][0-9]{0,8})$")

    @classmethod
    def parse(cls, s: str) -> "SuiteRef":
        m = cls._RE.fullmatch(s.strip()) if isinstance(s, str) else None
        if not m or not is_slug(m["name"]):
            raise InvalidReference(f"eval suite ref `{s}` must look like `name@vN`")
        return cls(m["name"], int(m["v"]))

    def __str__(self) -> str:
        return f"{self.name}@v{self.version}"


def _coerce(parser):
    def inner(v: Any) -> Any:
        if isinstance(v, str):
            return parser(v)
        return v

    return inner


DatasetRefField = Annotated[
    DatasetRef, BeforeValidator(_coerce(DatasetRef.parse)), PlainSerializer(str, return_type=str)
]
RecipeRefField = Annotated[RecipeRef, BeforeValidator(_coerce(RecipeRef.parse)), PlainSerializer(str, return_type=str)]
SuiteRefField = Annotated[SuiteRef, BeforeValidator(_coerce(SuiteRef.parse)), PlainSerializer(str, return_type=str)]


def new_job_id() -> str:
    return f"ftjob-{uuid.uuid7().hex if hasattr(uuid, 'uuid7') else _uuid7_hex()}"


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid7().hex if hasattr(uuid, 'uuid7') else _uuid7_hex()}"


def _uuid7_hex() -> str:
    """Fallback UUIDv7 (time-ordered) for Python < 3.14."""
    import os
    import time

    ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= ((rand >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= rand & ((1 << 62) - 1)
    return f"{value:032x}"


_JOB_ID = re.compile(r"^ftjob-[0-9a-f]{32}$")


def parse_job_id(s: str) -> str:
    if not _JOB_ID.fullmatch(s):
        raise InvalidReference(f"job id `{s}` is malformed")
    return s
