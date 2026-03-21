from pathlib import Path
import random

import cv2
import numpy as np


def _pick_base_image(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])
    for p in files:
        img = cv2.imread(str(p))
        if img is not None:
            return p, img
    return None, None


def _augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]

    # Random affine transform (rotation/scale/translation/shear-lite)
    angle = rng.uniform(-20.0, 20.0)
    scale = rng.uniform(0.82, 1.20)
    tx = rng.uniform(-0.18 * w, 0.18 * w)
    ty = rng.uniform(-0.18 * h, 0.18 * h)

    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty

    out = cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Occasional horizontal flip for additional variation.
    if rng.random() < 0.45:
        out = cv2.flip(out, 1)

    # Color jitter: brightness/contrast
    alpha = rng.uniform(0.78, 1.25)  # contrast
    beta = rng.uniform(-28, 28)      # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Mild blur/noise mix
    if rng.random() < 0.40:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    if rng.random() < 0.35:
        noise = np.random.normal(0, 6.5, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def regen_folder(folder: Path, count: int, seed: int = 42):
    base_path, base_img = _pick_base_image(folder)
    if base_img is None:
        print(f"[skip] {folder.name}: no readable base image")
        return False

    # Remove all existing files first.
    for p in folder.iterdir():
        if p.is_file():
            p.unlink()

    # Save canonical base.
    base_out = folder / f"{folder.name.lower()}_base.jpg"
    cv2.imwrite(str(base_out), base_img)

    rng = random.Random(seed + hash(folder.name) % 1000)
    for i in range(1, count + 1):
        aug = _augment(base_img, rng)
        out_path = folder / f"{folder.name.lower()}_{i:03d}.jpg"
        cv2.imwrite(str(out_path), aug)

    print(f"[ok] {folder.name}: kept 1 base + generated {count} images")
    return True


def main():
    root = Path("data/external/kaggle/indian-sign-language-dataset/ISL_Dataset")
    letters = ["H", "J", "Y"]
    target_per_letter = 60

    for letter in letters:
        folder = root / letter
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        regen_folder(folder, count=target_per_letter)


if __name__ == "__main__":
    main()
