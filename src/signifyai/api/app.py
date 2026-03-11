from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..data.dataset_version import DatasetVersionBuilder, DatasetVersionConfig
from ..model.registry import ModelRegistry
from ..model.sequence_model import SequenceModelPipeline, SequenceTrainConfig, load_runtime_model, predict_sequence_model
from ..nlp.intent_pack import INTENT_PACK, intent_text


class InferRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="2D sequence of flattened vectors")
    model_name: str | None = None


class BuildDatasetRequest(BaseModel):
    version: str


class TrainRequest(BaseModel):
    version: str
    model_name: str


class PromoteRequest(BaseModel):
    model_name: str
    notes: str = ""


def create_app() -> FastAPI:
    app = FastAPI(title="SignifyAI API", version="0.1.0")
    _enable_cors(app)
    _mount_web(app)

    registry = ModelRegistry()
    pipeline = SequenceModelPipeline()

    @app.get("/")
    def root() -> dict:
        return {"message": "SignifyAI API", "web": "/web"}

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "active_model": registry.active(),
            "intent_count": len(INTENT_PACK),
        }

    @app.get("/metrics")
    def metrics() -> dict:
        return {
            "latency_target_ms": 200,
            "active_model": registry.active(),
            "model_history": registry.history_count(),
        }

    @app.get("/intents")
    def intents() -> dict:
        return {"intents": INTENT_PACK}

    @app.post("/infer/stream")
    def infer_stream(req: InferRequest) -> dict:
        model_name = req.model_name or registry.active()
        if model_name is None:
            raise HTTPException(status_code=400, detail="No active model. Train and promote a model first.")

        model = load_runtime_model(model_name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        sequence = np.asarray(req.sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise HTTPException(status_code=400, detail="sequence must be 2D")

        label, confidence = predict_sequence_model(model, sequence)
        return {
            "intent_id": label,
            "intent_text": intent_text(label),
            "confidence": confidence,
            "source_model_version": model_name,
        }

    @app.post("/dataset/build")
    def build_dataset(req: BuildDatasetRequest) -> dict:
        builder = DatasetVersionBuilder(DatasetVersionConfig())
        return builder.build_dataset_version(req.version)

    @app.post("/train/sequence")
    def train_sequence(req: TrainRequest) -> dict:
        version_dir = Path("data/landmarks/versions") / req.version
        if not version_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset version not found: {req.version}")

        cfg = SequenceTrainConfig(version_dir=version_dir, model_name=req.model_name, out_dir=Path("data/models"))
        try:
            return pipeline.train_sequence_model(cfg)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

    @app.post("/train/promote")
    def promote(req: PromoteRequest) -> dict:
        return registry.promote_model(req.model_name, notes=req.notes)

    return app


def _enable_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _mount_web(app: FastAPI) -> None:
    web_dir = Path("web")
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")
