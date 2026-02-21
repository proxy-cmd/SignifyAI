from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_raw: Path
    data_processed: Path
    models: Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = Paths(
    project_root=PROJECT_ROOT,
    data_raw=PROJECT_ROOT / "data" / "raw",
    data_processed=PROJECT_ROOT / "data" / "processed",
    models=PROJECT_ROOT / "models",
)

DEFAULT_MODEL_PATH = PATHS.models / "gesture_model.joblib"
DEFAULT_LABELS_PATH = PATHS.models / "labels.json"
DEFAULT_DATASET_PATH = PATHS.data_processed / "dataset.csv"

LANDMARKS_PER_HAND = 21
LANDMARK_VALUES = 3
MAX_HANDS = 2
FEATURE_SIZE = LANDMARKS_PER_HAND * LANDMARK_VALUES * MAX_HANDS
