from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..data.dataset_version import DatasetVersionBuilder, DatasetVersionConfig
from ..model.registry import ModelRegistry
from ..model.sequence_model import (
    SequenceModelPipeline,
    SequenceTrainConfig,
    load_runtime_model,
    predict_sequence_model,
)
from ..nlp.intent_pack import INTENT_PACK, intent_text


class InferRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="Flattened sequence vectors")
    model_name: str | None = None


class SessionStartRequest(BaseModel):
    intent_id: str
    signer_id: str = "anonymous"
    consent_raw_video: bool = False


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    registry = ModelRegistry()
    pipeline = SequenceModelPipeline()

    web_dir = Path("web")
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "active_model": registry.active(), "intent_count": len(INTENT_PACK)}

    @app.get("/metrics")
    def metrics() -> dict:
        return {
            "latency_target_ms": 200,
            "active_model": registry.active(),
            "model_history": len((registry._load()).get("history", [])),
        }

    @app.get("/intents")
    def intents() -> dict:
        return {"intents": INTENT_PACK}

    @app.post("/infer/stream")
    def infer_stream(req: InferRequest) -> dict:
        model_name = req.model_name or registry.active()
        if model_name is None:
            raise HTTPException(status_code=400, detail="No promoted model available")
        model = load_runtime_model(model_name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        seq = np.asarray(req.sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise HTTPException(status_code=400, detail="sequence must be 2D")
        lbl, conf = predict_sequence_model(model, seq)
        return {
            "intent_id": lbl,
            "intent_text": intent_text(lbl),
            "confidence": conf,
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
        try:
            return pipeline.train_sequence_model(
                SequenceTrainConfig(
                    version_dir=version_dir,
                    model_name=req.model_name,
                    out_dir=Path("data/models"),
                )
            )
        except Exception as ex:
            raise HTTPException(status_code=400, detail=str(ex))

    @app.post("/train/promote")
    def promote(req: PromoteRequest) -> dict:
        return registry.promote_model(req.model_name, notes=req.notes)

    @app.get("/")
    def root() -> dict:
        return {"message": "SignifyAI API", "web": "/web"}

    return app
