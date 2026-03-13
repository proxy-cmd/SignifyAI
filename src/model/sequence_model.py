from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from dataset.dataset_builder import load_split_xy
from .evaluation import EvalOut


class SeqCfg:
    def __init__(self, version_dir, model_name, out_dir=Path("data/models"), seq_len=24):
        self.version_dir = version_dir
        self.model_name = model_name
        self.out_dir = out_dir
        self.seq_len = seq_len


def make_model_list():
    # keep it simple: one list with model object + whether scaling is needed
    model_list = []
    model_list.append({"name": "logreg", "model": LogisticRegression(max_iter=3000, class_weight="balanced"), "scale": True})
    model_list.append(
        {
            "name": "linear_svc_cal",
            "model": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=3),
            "scale": True,
        }
    )
    model_list.append({"name": "rbf_svc", "model": SVC(kernel="rbf", probability=True, class_weight="balanced"), "scale": True})
    model_list.append(
        {
            "name": "random_forest",
            "model": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "scale": False,
        }
    )
    model_list.append(
        {
            "name": "extra_trees",
            "model": ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "scale": False,
        }
    )
    model_list.append({"name": "knn", "model": KNeighborsClassifier(n_neighbors=3), "scale": True})
    model_list.append(
        {
            "name": "mlp",
            "model": MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=800,
                alpha=1e-4,
                learning_rate_init=1e-3,
                random_state=42,
            ),
            "scale": True,
        }
    )
    return model_list


def fit_bundle(bundle, x_train, y_train):
    # bundle = {"name","model","scale","scaler"}
    if bundle["scale"]:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        bundle["scaler"] = scaler
        bundle["model"].fit(x_scaled, y_train)
    else:
        bundle["scaler"] = None
        bundle["model"].fit(x_train, y_train)
    return bundle


def predict_bundle(bundle, x):
    x_input = x
    if bundle.get("scale") and bundle.get("scaler") is not None:
        x_input = bundle["scaler"].transform(x)
    return bundle["model"].predict(x_input)


def predict_proba_bundle(bundle, x):
    x_input = x
    if bundle.get("scale") and bundle.get("scaler") is not None:
        x_input = bundle["scaler"].transform(x)

    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_input)

    if hasattr(model, "decision_function"):
        score = model.decision_function(x_input)
        if score.ndim == 1:
            score = np.expand_dims(score, axis=1)
        score = score - np.max(score, axis=1, keepdims=True)
        exp_score = np.exp(score)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        return probs

    # fallback to one-hot
    pred = model.predict(x_input)
    cls = list(getattr(model, "classes_", []))
    if not cls:
        return np.zeros((len(pred), 1), dtype=np.float32)
    out = np.zeros((len(pred), len(cls)), dtype=np.float32)
    for i in range(len(pred)):
        label = pred[i]
        if label in cls:
            j = cls.index(label)
            out[i, j] = 1.0
    return out


def score_bundle(bundle, x, y):
    if x.shape[0] == 0:
        return 0.0
    pred = predict_bundle(bundle, x)
    return float(accuracy_score(y, pred))


def check_train_data(x_train, y_train):
    if x_train.shape[0] < 2:
        raise ValueError("Not enough training samples. Build dataset and record more clips.")

    labels = set(y_train.tolist())
    if len(labels) < 2:
        only = ", ".join(sorted(str(v) for v in labels))
        raise ValueError(f"Need at least 2 intent labels in training split. Found: {only or 'none'}")


class SeqTrainer:
    def train(self, cfg):
        # step 1: load data
        x_train, y_train = load_split_xy(cfg.version_dir, "train", seq_len=cfg.seq_len)
        x_val, y_val = load_split_xy(cfg.version_dir, "val", seq_len=cfg.seq_len)
        x_test, y_test = load_split_xy(cfg.version_dir, "test", seq_len=cfg.seq_len)
        check_train_data(x_train, y_train)

        # step 2: train all candidates
        rows = []
        best_name = ""
        best_bundle = None
        best_val = -1.0
        model_list = make_model_list()

        for item in model_list:
            name = item["name"]
            model = item["model"]
            use_scale = item["scale"]
            bundle = {"name": name, "model": model, "scale": use_scale, "scaler": None}
            try:
                bundle = fit_bundle(bundle, x_train, y_train)
                train_acc = score_bundle(bundle, x_train, y_train)
                val_acc = score_bundle(bundle, x_val, y_val)
                test_acc = score_bundle(bundle, x_test, y_test)
                rows.append(
                    {
                        "model": name,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "test_accuracy": test_acc,
                    }
                )
                if val_acc > best_val:
                    best_val = val_acc
                    best_name = name
                    best_bundle = bundle
            except Exception as ex:
                rows.append({"model": name, "error": str(ex)})

        if best_bundle is None:
            raise ValueError("All model candidates failed. Check dataset quality and label balance.")

        # step 3: save best model
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        model_path = cfg.out_dir / f"{cfg.model_name}.joblib"
        meta_path = cfg.out_dir / f"{cfg.model_name}.json"
        joblib.dump(best_bundle, model_path)

        # step 4: save metadata
        test_acc = score_bundle(best_bundle, x_test, y_test)
        cls = list(getattr(best_bundle["model"], "classes_", []))
        class_list = []
        for c in cls:
            class_list.append(str(c))

        meta = {
            "model_name": cfg.model_name,
            "version_dir": str(cfg.version_dir),
            "seq_len": int(cfg.seq_len),
            "best_algo": best_name,
            "val_accuracy": float(best_val),
            "test_accuracy": test_acc,
            "classes": class_list,
            "leaderboard": rows,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        label_list = sorted(str(v) for v in set(y_train.tolist()))
        return {
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "best_algo": best_name,
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "labels": label_list,
            "val_accuracy": float(best_val),
            "test_accuracy": test_acc,
            "leaderboard": rows,
        }

    def eval(self, version_dir, model_name, out_dir=Path("data/models")):
        model_path = out_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        seq_len = 24
        meta_path = out_dir / f"{model_name}.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                seq_len = int(meta.get("seq_len", 24))
            except Exception:
                seq_len = 24

        bundle = joblib.load(model_path)
        x_test, y_test = load_split_xy(version_dir, "test", seq_len=seq_len)
        if x_test.shape[0] == 0:
            msg = "No test samples. Build dataset again with more clips per label."
            return EvalOut(acc=0.0, report=msg, samples=0)

        pred = predict_bundle(bundle, x_test)
        acc = float(accuracy_score(y_test, pred))
        report = str(classification_report(y_test, pred, zero_division=0))
        return EvalOut(acc=acc, report=report, samples=int(x_test.shape[0]))


def load_model_for_runtime(model_name, out_dir=Path("data/models")):
    path = out_dir / f"{model_name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict_seq(model_bundle, seq_matrix):
    if model_bundle is None:
        return "unknown", 0.0

    flat = seq_matrix.reshape(1, -1)
    probs = predict_proba_bundle(model_bundle, flat)[0]
    idx = int(np.argmax(probs))
    cls = list(getattr(model_bundle["model"], "classes_", []))
    if not cls:
        return "unknown", 0.0
    label = str(cls[idx])
    conf = float(probs[idx])
    return label, conf
