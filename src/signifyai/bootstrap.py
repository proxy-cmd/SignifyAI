from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_RAW_IMAGES_DIR,
)
from .external_data import import_dataset_from_url, import_from_kaggle
from .image_dataset import BuildImageDatasetConfig, build_dataset_from_images
from .train import TrainConfig, run_training


@dataclass
class BootstrapConfig:
    kaggle_slug: str = "grassknoted/asl-alphabet"
    images_dir: Path = DEFAULT_RAW_IMAGES_DIR
    dataset_csv: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    max_per_class: int = 1200
    min_free_gb: float = 20.0


@dataclass
class BootstrapURLConfig:
    dataset_url: str
    images_dir: Path = DEFAULT_RAW_IMAGES_DIR
    dataset_csv: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    max_per_class: int = 1200
    min_free_gb: float = 20.0


def _ensure_free_space(min_free_gb: float, path: Path) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough free disk space. Free: {free_gb:.1f} GB, required: {min_free_gb:.1f} GB."
        )


def run_bootstrap(cfg: BootstrapConfig) -> None:
    _ensure_free_space(cfg.min_free_gb, cfg.images_dir.parent)
    print(f"[BOOTSTRAP] Importing dataset from Kaggle: {cfg.kaggle_slug}")
    target = import_from_kaggle(cfg.kaggle_slug, cfg.images_dir, force=False)
    print(f"[BOOTSTRAP] Dataset ready: {target}")

    print("[BOOTSTRAP] Building landmark CSV from images...")
    total, saved = build_dataset_from_images(
        BuildImageDatasetConfig(
            root_dir=cfg.images_dir,
            out_csv=cfg.dataset_csv,
            max_images_per_class=cfg.max_per_class,
            min_detection_confidence=0.55,
        )
    )
    print(f"[BOOTSTRAP] Processed images: {total}, saved samples: {saved}")

    print("[BOOTSTRAP] Training AutoML model...")
    run_training(
        TrainConfig(
            dataset_csv=cfg.dataset_csv,
            model_path=cfg.model_path,
            labels_path=cfg.labels_path,
            metadata_path=cfg.metadata_path,
            automl=True,
        )
    )
    print("[BOOTSTRAP] Done.")


def run_bootstrap_from_url(cfg: BootstrapURLConfig) -> None:
    _ensure_free_space(cfg.min_free_gb, cfg.images_dir.parent)
    print(f"[BOOTSTRAP] Importing dataset from URL: {cfg.dataset_url}")
    extracted = import_dataset_from_url(cfg.dataset_url, cfg.images_dir)
    print(f"[BOOTSTRAP] Extracted files: {extracted}")

    print("[BOOTSTRAP] Building landmark CSV from images...")
    total, saved = build_dataset_from_images(
        BuildImageDatasetConfig(
            root_dir=cfg.images_dir,
            out_csv=cfg.dataset_csv,
            max_images_per_class=cfg.max_per_class,
            min_detection_confidence=0.55,
        )
    )
    print(f"[BOOTSTRAP] Processed images: {total}, saved samples: {saved}")

    print("[BOOTSTRAP] Training AutoML model...")
    run_training(
        TrainConfig(
            dataset_csv=cfg.dataset_csv,
            model_path=cfg.model_path,
            labels_path=cfg.labels_path,
            metadata_path=cfg.metadata_path,
            automl=True,
        )
    )
    print("[BOOTSTRAP] Done.")
