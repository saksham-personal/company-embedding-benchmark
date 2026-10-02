from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from urllib.parse import urlparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


class DownloadError(RuntimeError):
    pass


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        redirected = super().redirect_request(request, file, code, message, headers, new_url)
        if redirected is not None and urlparse(new_url).netloc != urlparse(request.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(url: str, destination: str | Path, expected_sha256: str) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        actual = sha256_file(destination)
        if actual.lower() == expected_sha256.lower():
            return destination
        raise DownloadError(f"cached file has wrong SHA-256: {destination}; remove it or use a clean work directory")
    request = urllib.request.Request(url, headers={"User-Agent": "company-embedding-benchmark/0.1"})
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and urlparse(url).hostname in {"github.com", "api.github.com"}:
        request.add_header("Authorization", f"Bearer {token}")
    resume_from = partial.stat().st_size if partial.exists() else 0
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
    try:
        with urllib.request.build_opener(_SafeRedirect()).open(request, timeout=120) as response:
            append = resume_from > 0 and getattr(response, "status", None) == 206
            with partial.open("ab" if append else "wb") as handle:
                if resume_from and not append:
                    resume_from = 0
                shutil.copyfileobj(response, handle, length=1024 * 1024)
    except Exception as exc:
        raise DownloadError(f"download failed for {url}: {exc}") from exc
    actual = sha256_file(partial)
    if actual.lower() != expected_sha256.lower():
        partial.unlink(missing_ok=True)
        raise DownloadError(f"SHA-256 mismatch for {url}: {actual} != {expected_sha256}")
    partial.replace(destination)
    return destination


def safe_extract_zip(archive: str | Path, destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            parts = PurePosixPath(member.filename).parts
            if (not parts or member.filename.startswith(("/", "\\")) or ".." in parts
                    or "\\" in member.filename or ":" in member.filename or "\x00" in member.filename):
                raise DownloadError(f"unsafe zip member: {member.filename}")
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise DownloadError(f"zip symlink is not allowed: {member.filename}")
            target = (destination / Path(*parts)).resolve()
            if destination not in target.parents and target != destination:
                raise DownloadError(f"zip member escapes destination: {member.filename}")
        handle.extractall(destination)
    return destination


def verify_payload(root: str | Path, files: list[dict[str, object]]) -> None:
    root = Path(root).resolve()
    for record in files:
        relative = Path(str(record["path"]))
        path = (root / relative).resolve()
        if root not in path.parents:
            raise DownloadError(f"payload path escapes install root: {relative}")
        if not path.is_file():
            raise DownloadError(f"payload file is missing: {relative}")
        expected_size = int(record["size"])
        if path.stat().st_size != expected_size:
            raise DownloadError(f"payload size mismatch: {relative}")
        actual = sha256_file(path)
        if actual.lower() != str(record["sha256"]).lower():
            raise DownloadError(f"payload SHA-256 mismatch: {relative}")

