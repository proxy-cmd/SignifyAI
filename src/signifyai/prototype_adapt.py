from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
from google.protobuf import message_factory, symbol_database
import mediapipe as mp
import numpy as np

from .config import DEFAULT_PROTOTYPE_DB_PATH, FEATURE_SIZE, LANDMARKS_PER_HAND, MAX_HANDS
from .feature_extraction import normalize_features
from .phrase_map import set_phrase


def _patch_protobuf_for_mediapipe() -> None:
    if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        def _symbol_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        symbol_database.SymbolDatabase.GetPrototype = _symbol_get_prototype  # type: ignore[attr-defined]

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        def _factory_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        message_factory.MessageFactory.GetPrototype = _factory_get_prototype  # type: ignore[attr-defined]


_patch_protobuf_for_mediapipe()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PrototypeDB:
    vectors: np.ndarray  # [N, F]
    labels: np.ndarray  # [N]
    norms: np.ndarray  # [N, F]


@dataclass
class PrototypeMatch:
    label: str
    similarity: float
    margin: float


@dataclass
class ImportStats:
    total_images: int
    detected_images: int
    saved_vectors: int
    labels_added: list[str]


@dataclass
class ImagePointsResult:
    hand_count: int
    best_variant: str
    out_image: Optional[Path]


def _iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _enhance_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [("orig", image)]

    up = cv2.resize(image, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale", up))

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    variants.append(("equalized", cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)))

    blur = cv2.GaussianBlur(image, (0, 0), 1.1)
    sharp = cv2.addWeighted(image, 1.25, blur, -0.25, 0)
    variants.append(("sharpen", sharp))

    den = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    variants.append(("denoise", den))
    return variants


def _results_to_features(results) -> tuple[np.ndarray, int, float]:
    features = np.zeros((FEATURE_SIZE,), dtype=np.float32)
    if not results or not results.multi_hand_landmarks:
        return features, 0, 0.0

    slot_to_features: dict[int, np.ndarray] = {}
    hand_area_best = 0.0
    handedness = results.multi_handedness or []
    hand_size = LANDMARKS_PER_HAND * 3

    for i, lm in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
        slot = i
        if i < len(handedness):
            label = handedness[i].classification[0].label.lower()
            if label == "left":
                slot = 0
            elif label == "right":
                slot = 1

        arr = np.asarray([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        slot_to_features[slot] = arr.flatten()
        xs = arr[:, 0]
        ys = arr[:, 1]
        hand_area_best = max(hand_area_best, float((xs.max() - xs.min()) * (ys.max() - ys.min())))

    for slot in range(MAX_HANDS):
        start = slot * hand_size
        end = start + hand_size
        features[start:end] = slot_to_features.get(slot, np.zeros((hand_size,), dtype=np.float32))

    return normalize_features(features), len(slot_to_features), hand_area_best


def extract_points_from_image(
    image_path: Path,
    min_detection_confidence: float = 0.35,
    min_tracking_confidence: float = 0.30,
    save_overlay_to: Optional[Path] = None,
) -> tuple[Optional[np.ndarray], ImagePointsResult]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=1,
    )
    try:
        return _extract_points_from_image_array(image, hands, save_overlay_to=save_overlay_to)
    finally:
        hands.close()


def _extract_points_from_image_array(
    image: np.ndarray,
    hands,
    save_overlay_to: Optional[Path] = None,
) -> tuple[Optional[np.ndarray], ImagePointsResult]:
    mp_draw = mp.solutions.drawing_utils

    best_features: Optional[np.ndarray] = None
    best_hand_count = 0
    best_score = -1.0
    best_variant = "none"
    best_results = None

    for name, variant in _enhance_variants(image):
        rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        feats, hand_count, area = _results_to_features(res)
        score = float(hand_count) * 2.0 + area
        if hand_count > 0 and score > best_score:
            best_score = score
            best_features = feats
            best_hand_count = hand_count
            best_variant = name
            best_results = res

    out_img_path: Optional[Path] = None
    if save_overlay_to is not None:
        show = image.copy()
        if best_results and best_results.multi_hand_landmarks:
            for lm in best_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(show, lm, mp.solutions.hands.HAND_CONNECTIONS)
        save_overlay_to.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_overlay_to), show)
        out_img_path = save_overlay_to

    return best_features, ImagePointsResult(
        hand_count=best_hand_count,
        best_variant=best_variant,
        out_image=out_img_path,
    )


def load_prototype_db(path: Path = DEFAULT_PROTOTYPE_DB_PATH) -> PrototypeDB:
    if not path.exists():
        zeros = np.zeros((0, FEATURE_SIZE), dtype=np.float32)
        labels = np.asarray([], dtype=str)
        return PrototypeDB(vectors=zeros, labels=labels, norms=zeros)
    data = np.load(path, allow_pickle=False)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=str)
    if vectors.ndim != 2 or vectors.shape[1] != FEATURE_SIZE:
        raise ValueError(f"Invalid prototype vectors shape: {vectors.shape}")
    if labels.ndim != 1 or labels.shape[0] != vectors.shape[0]:
        raise ValueError("Invalid prototype labels shape")
    norms = _l2_normalize_rows(vectors)
    return PrototypeDB(vectors=vectors, labels=labels, norms=norms)


def save_prototype_db(db: PrototypeDB, path: Path = DEFAULT_PROTOTYPE_DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vectors=db.vectors.astype(np.float32), labels=db.labels.astype(str))


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (x / denom).astype(np.float32)


def append_prototypes(
    vectors: list[np.ndarray],
    labels: list[str],
    path: Path = DEFAULT_PROTOTYPE_DB_PATH,
    dedupe_similarity: float = 0.995,
) -> int:
    if not vectors:
        return 0
    if len(vectors) != len(labels):
        raise ValueError("vectors/labels length mismatch")

    db = load_prototype_db(path)
    current_vectors = db.vectors
    current_labels = db.labels
    current_norms = db.norms

    added = 0
    new_vectors: list[np.ndarray] = []
    new_labels: list[str] = []
    for vec, label in zip(vectors, labels):
        v = np.asarray(vec, dtype=np.float32).reshape(FEATURE_SIZE)
        if np.allclose(v, 0.0):
            continue
        v_norm = (v / max(float(np.linalg.norm(v)), 1e-8)).astype(np.float32)
        if current_norms.shape[0] > 0:
            sims = current_norms @ v_norm
            same_label = current_labels == label
            if np.any(same_label):
                max_same = float(np.max(sims[same_label]))
                if max_same >= dedupe_similarity:
                    continue
        new_vectors.append(v)
        new_labels.append(str(label))
        added += 1

    if added == 0:
        return 0

    all_vectors = np.vstack([current_vectors, np.asarray(new_vectors, dtype=np.float32)]) if current_vectors.size else np.asarray(new_vectors, dtype=np.float32)
    all_labels = np.concatenate([current_labels, np.asarray(new_labels, dtype=str)]) if current_labels.size else np.asarray(new_labels, dtype=str)
    all_db = PrototypeDB(vectors=all_vectors.astype(np.float32), labels=all_labels.astype(str), norms=_l2_normalize_rows(all_vectors.astype(np.float32)))
    save_prototype_db(all_db, path)
    return added


def predict_prototype(
    features: np.ndarray,
    db: PrototypeDB,
    min_similarity: float = 0.84,
    min_margin: float = 0.03,
) -> Optional[PrototypeMatch]:
    if db.vectors.shape[0] == 0:
        return None
    vec = np.asarray(features, dtype=np.float32).reshape(FEATURE_SIZE)
    if np.allclose(vec, 0.0):
        return None
    vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
    sims = db.norms @ vec
    top_idx = np.argsort(sims)[::-1]
    i0 = int(top_idx[0])
    s0 = float(sims[i0])
    s1 = float(sims[top_idx[1]]) if len(top_idx) > 1 else -1.0
    margin = s0 - s1
    if s0 < min_similarity or margin < min_margin:
        return None
    return PrototypeMatch(label=str(db.labels[i0]), similarity=s0, margin=margin)


def adapt_sign_from_images(
    label: str,
    image_paths: list[Path],
    out_db: Path = DEFAULT_PROTOTYPE_DB_PATH,
    min_detection_confidence: float = 0.35,
    phrase_text: Optional[str] = None,
) -> ImportStats:
    vectors: list[np.ndarray] = []
    total = 0
    detected = 0
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.30,
        model_complexity=1,
    )
    try:
        for p in image_paths:
            total += 1
            image = cv2.imread(str(p))
            if image is None:
                continue
            feats, info = _extract_points_from_image_array(image, hands, save_overlay_to=None)
            if feats is None or info.hand_count == 0:
                continue
            vectors.append(feats.astype(np.float32))
            detected += 1
    finally:
        hands.close()

    added = append_prototypes(vectors=vectors, labels=[label] * len(vectors), path=out_db)
    if phrase_text:
        set_phrase(label, phrase_text)
    return ImportStats(
        total_images=total,
        detected_images=detected,
        saved_vectors=added,
        labels_added=[label] if added > 0 else [],
    )


def adapt_signs_from_folder(
    images_root: Path,
    out_db: Path = DEFAULT_PROTOTYPE_DB_PATH,
    max_per_label: int = 0,
    min_detection_confidence: float = 0.35,
) -> ImportStats:
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    total = 0
    detected = 0
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    labels_added_set: set[str] = set()

    class_dirs = [p for p in sorted(images_root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not class_dirs:
        raise ValueError("Expected subfolders by label under images_root.")

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.30,
        model_complexity=1,
    )
    try:
        for class_dir in class_dirs:
            label = class_dir.name.strip().lower().replace(" ", "_")
            used = 0
            for image_path in _iter_images(class_dir):
                if max_per_label > 0 and used >= max_per_label:
                    break
                total += 1
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                feats, info = _extract_points_from_image_array(image, hands, save_overlay_to=None)
                if feats is None or info.hand_count == 0:
                    continue
                vectors.append(feats.astype(np.float32))
                labels.append(label)
                used += 1
                detected += 1
                labels_added_set.add(label)
    finally:
        hands.close()

    added = append_prototypes(vectors=vectors, labels=labels, path=out_db)
    return ImportStats(
        total_images=total,
        detected_images=detected,
        saved_vectors=added,
        labels_added=sorted(labels_added_set),
    )
