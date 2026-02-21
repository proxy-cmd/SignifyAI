from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = out_dir / member.filename
            if not str(member_path.resolve()).startswith(str(out_dir.resolve())):
                continue
            zf.extract(member, out_dir)
            extracted += 1
    return extracted


def download_from_url(url: str, out_path: Path, timeout_sec: int = 120) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout_sec) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return out_path


def import_zip_dataset(zip_file: Path, target_dir: Path, cleanup_zip: bool = False) -> int:
    if not zip_file.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_file}")
    count = _safe_extract_zip(zip_file, target_dir)
    if cleanup_zip:
        zip_file.unlink(missing_ok=True)
    return count


def import_dataset_from_url(url: str, target_dir: Path) -> int:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "dataset.zip"
    if not name.endswith(".zip"):
        name += ".zip"
    tmp_zip = target_dir.parent / name
    download_from_url(url, tmp_zip)
    return import_zip_dataset(tmp_zip, target_dir, cleanup_zip=True)


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
