from __future__ import annotations

import ipaddress
import shutil
import zipfile
from typing import Final
from pathlib import Path
from urllib.parse import urlparse

import requests


MAX_EXTRACT_FILES: Final[int] = 100_000
MAX_MEMBER_BYTES: Final[int] = 256 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES: Final[int] = 2 * 1024 * 1024 * 1024
MAX_DOWNLOAD_BYTES: Final[int] = 1536 * 1024 * 1024


def _is_private_or_local_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    if not host:
        return True
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    if host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed.")
    if not parsed.netloc:
        raise ValueError("URL must include a hostname.")
    if _is_private_or_local_host(parsed.hostname or ""):
        raise ValueError("Refusing local/private host URL for security.")


def _safe_extract_zip(
    zip_path: Path,
    out_dir: Path,
    *,
    max_files: int = MAX_EXTRACT_FILES,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_total_uncompressed_bytes: int = MAX_TOTAL_UNCOMPRESSED_BYTES,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_resolved = out_dir.resolve()
    extracted = 0
    total_uncompressed = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if extracted >= max_files:
                raise RuntimeError(f"ZIP extraction aborted: file count limit reached ({max_files}).")

            member_name = member.filename.replace("\\", "/")
            member_path = Path(member_name)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue

            target = (out_dir / member_path).resolve()
            try:
                target.relative_to(out_dir_resolved)
            except ValueError:
                continue

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if member.file_size > max_member_bytes:
                raise RuntimeError(
                    f"ZIP extraction aborted: member too large ({member.filename}, {member.file_size} bytes)."
                )
            total_uncompressed += int(member.file_size)
            if total_uncompressed > max_total_uncompressed_bytes:
                raise RuntimeError(
                    f"ZIP extraction aborted: total uncompressed size exceeds limit ({max_total_uncompressed_bytes} bytes)."
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted += 1
    return extracted


def download_from_url(
    url: str,
    out_path: Path,
    timeout_sec: int = 120,
    max_download_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Path:
    _validate_remote_url(url)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout_sec) as resp:
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length is not None:
            try:
                advertised = int(content_length)
            except ValueError:
                advertised = -1
            if advertised > max_download_bytes > 0:
                raise RuntimeError(
                    f"Download exceeds limit: {advertised} bytes > {max_download_bytes} bytes."
                )
        with open(out_path, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_download_bytes > 0:
                        raise RuntimeError(
                            f"Download aborted: size exceeds limit ({max_download_bytes} bytes)."
                        )
                    f.write(chunk)

    return out_path


def import_zip_dataset(
    zip_file: Path,
    target_dir: Path,
    cleanup_zip: bool = False,
    *,
    max_files: int = MAX_EXTRACT_FILES,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_total_uncompressed_bytes: int = MAX_TOTAL_UNCOMPRESSED_BYTES,
) -> int:
    if not zip_file.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_file}")
    count = _safe_extract_zip(
        zip_file,
        target_dir,
        max_files=max_files,
        max_member_bytes=max_member_bytes,
        max_total_uncompressed_bytes=max_total_uncompressed_bytes,
    )
    if cleanup_zip:
        zip_file.unlink(missing_ok=True)
    return count


def import_dataset_from_url(
    url: str,
    target_dir: Path,
    *,
    timeout_sec: int = 120,
    max_download_bytes: int = MAX_DOWNLOAD_BYTES,
    max_files: int = MAX_EXTRACT_FILES,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_total_uncompressed_bytes: int = MAX_TOTAL_UNCOMPRESSED_BYTES,
) -> int:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "dataset.zip"
    if not name.endswith(".zip"):
        name += ".zip"
    tmp_zip = target_dir.parent / name
    download_from_url(url, tmp_zip, timeout_sec=timeout_sec, max_download_bytes=max_download_bytes)
    return import_zip_dataset(
        tmp_zip,
        target_dir,
        cleanup_zip=True,
        max_files=max_files,
        max_member_bytes=max_member_bytes,
        max_total_uncompressed_bytes=max_total_uncompressed_bytes,
    )


def import_from_kaggle(dataset_slug: str, target_dir: Path, force: bool = False) -> Path:
    """Download a Kaggle dataset using kagglehub if available.

    Example slug: "grassknoted/asl-alphabet"
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub  # type: ignore
    except Exception as ex:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "kagglehub is not installed. Install with: pip install kagglehub"
        ) from ex

    cache_path = Path(kagglehub.dataset_download(dataset_slug, force_download=force))

    # Copy dataset to project target_dir for stable project-local path.
    if cache_path.is_dir():
        for item in cache_path.iterdir():
            dst = target_dir / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        return target_dir

    if cache_path.suffix.lower() == ".zip":
        import_zip_dataset(cache_path, target_dir)
        return target_dir

    raise RuntimeError(f"Unsupported Kaggle download artifact: {cache_path}")
