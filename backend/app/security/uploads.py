"""Untrusted-upload handling: size limits, type/content validation, safe parsing, safe paths.

Uploaded files are only ever parsed as CSV text by pandas. They are never executed, imported or
unpickled, and the stored copy is a re-serialised CSV under a server-generated file name.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from ..core.config import get_settings

ALLOWED_EXTENSIONS = {".csv"}
BINARY_SIGNATURES = (b"PK\x03\x04", b"%PDF", b"MZ", b"\x7fELF", b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"\x1f\x8b",
                     b"SQLite format 3", b"\xd0\xcf\x11\xe0", b"\x80\x04\x95", b"\x80\x03", b"PAR1", b"ARROW1")
CHUNK = 1024 * 1024
_NAME_BAD = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message, self.status = message, status


@dataclass
class ParsedUpload:
    df: pd.DataFrame
    filename: str
    size_bytes: int
    sha256: str


def sanitize_filename(name: str | None) -> str:
    base = Path((name or "upload.csv").replace("\\", "/")).name          # drop any directory components
    base = _NAME_BAD.sub("_", base).strip("._") or "upload.csv"
    return base[:100]


def safe_join(base: Path, *parts: str) -> Path:
    """Join and confirm the result stays inside ``base`` (path traversal protection)."""
    base = base.resolve()
    target = base.joinpath(*parts).resolve()
    if base != target and base not in target.parents:
        raise UploadError("invalid path", 400)
    return target


async def read_limited(file: UploadFile, limit: int) -> bytes:
    buf = io.BytesIO()
    total = 0
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise UploadError(f"file exceeds the {limit // (1024 * 1024)} MB upload limit", 413)
        buf.write(chunk)
    return buf.getvalue()


def validate_and_parse(data: bytes, filename: str | None) -> ParsedUpload:
    s = get_settings()
    name = sanitize_filename(filename)
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadError("only .csv files are accepted", 415)
    if not data.strip():
        raise UploadError("the file is empty")
    head = data[:16]
    if any(head.startswith(sig) for sig in BINARY_SIGNATURES) or b"\x00" in data[:65536]:
        raise UploadError("the file content is binary, not CSV text", 415)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadError("the file must be UTF-8 encoded text") from exc
    try:
        reader = csv.reader(io.StringIO(text))
        header = next(reader, [])
        for line_no, row in enumerate(reader, start=2):
            if line_no > s.max_rows + 2:
                break
            if row and len(row) != len(header):
                raise UploadError(f"row {line_no} has {len(row)} fields but the header has {len(header)}")
    except csv.Error as exc:
        raise UploadError(f"the file is not valid CSV: {str(exc)[:160]}") from exc
    hdr = [h.strip() for h in header]
    if len(set(hdr)) != len(hdr):
        raise UploadError("column names must be unique")
    try:
        df = pd.read_csv(io.StringIO(text), nrows=s.max_rows + 1, on_bad_lines="error", skipinitialspace=True, index_col=False,
                         low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise UploadError("the file contains no data") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise UploadError(f"the file is not valid CSV: {str(exc)[:160]}") from exc
    if len(df) > s.max_rows:
        raise UploadError(f"too many rows (limit {s.max_rows:,})", 413)
    if df.shape[1] < 2:
        raise UploadError("the file needs at least two columns (features and a label or prediction)")
    if df.shape[1] > s.max_columns:
        raise UploadError(f"too many columns (limit {s.max_columns})", 413)
    if len(df) < 1:
        raise UploadError("the file has a header but no rows")
    cols = [str(c).strip() for c in df.columns]
    if any(not c or c.lower().startswith("unnamed:") for c in cols):
        raise UploadError("every column needs a non-empty header name")
    if len(set(cols)) != len(cols):
        raise UploadError("column names must be unique")
    if any(len(c) > 100 for c in cols):
        raise UploadError("column names are limited to 100 characters")
    df.columns = cols
    return ParsedUpload(df, name, len(data), hashlib.sha256(data).hexdigest())


def describe_schema(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        item = {"name": c, "dtype": "numeric" if numeric else "categorical", "null_rate": float(s.isna().mean()),
                "n_unique": int(s.nunique(dropna=True)),
                "sample": [None if pd.isna(v) else (float(v) if numeric else str(v)[:40]) for v in s.dropna().head(3)]}
        if numeric and s.notna().any():
            item.update(min=float(s.min()), max=float(s.max()), mean=float(s.mean()))
        out.append(item)
    return out


def store_dataframe(df: pd.DataFrame, project_id: str, role: str) -> tuple[str, Path]:
    s = get_settings()
    folder = safe_join(s.data_dir, "projects", project_id)
    folder.mkdir(parents=True, exist_ok=True)
    rel = f"projects/{project_id}/{role}-{uuid.uuid4().hex}.csv"
    path = safe_join(s.data_dir, rel)
    df.to_csv(path, index=False)
    return rel, path


def resolve_storage(rel: str) -> Path:
    return safe_join(get_settings().data_dir, rel)


def remove_storage(rel: str) -> None:
    try:
        resolve_storage(rel).unlink(missing_ok=True)
    except (OSError, UploadError):
        pass


def csv_safe(value):
    """Neutralise spreadsheet formula injection in exported CSV cells."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value
