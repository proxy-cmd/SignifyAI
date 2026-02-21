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
DEFAULT_SESSION_LOG_PATH = PATHS.data_processed / "session_log.csv"
DEFAULT_REPORT_PATH = PATHS.data_processed / "session_report.md"
DEFAULT_CONFUSION_CSV_PATH = PATHS.data_processed / "confusion_matrix.csv"
DEFAULT_SEQUENCE_DATASET_PATH = PATHS.data_processed / "sequence_dataset.npz"
DEFAULT_PHRASE_MAP_PATH = PATHS.data_processed / "custom_phrases.json"
DEFAULT_METADATA_PATH = PATHS.models / "model_metadata.json"
DEFAULT_RAW_IMAGES_DIR = PATHS.data_raw / "images"
DEFAULT_TEMPORAL_MODEL_PATH = PATHS.models / "temporal_gesture_model.joblib"
DEFAULT_TEMPORAL_LABELS_PATH = PATHS.models / "temporal_labels.json"
DEFAULT_TEMPORAL_METADATA_PATH = PATHS.models / "temporal_model_metadata.json"

LANDMARKS_PER_HAND = 21
LANDMARK_VALUES = 3
MAX_HANDS = 2
FEATURE_SIZE = LANDMARKS_PER_HAND * LANDMARK_VALUES * MAX_HANDS
