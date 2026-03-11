# SignifyAI

SignifyAI is an ISL-focused assistive communication system designed for continuous sign-to-speech translation.

## Current Implementation (Vertical Slice)

- Streaming runtime with stage timing probes (`capture`, `perception`, `decode`, `speech`, `render`)
- Baseline intent decoder for task-focused phrase pack
- Landmark-first recording workflow with quality reports
- Dataset version builder with signer-aware split manifests
- Sequence model training/evaluation/promote pipeline
- Model registry with active/promoted versions
- FastAPI service with health/metrics/infer/dataset/train endpoints
- Basic web dashboard for health + metrics + intents

## Quick Start

```bash
python -m pip install -r requirements.txt
python -u .\run_this.py
```

## CLI

```bash
python -u .\src\main.py run
python -u .\src\main.py record --intent hospital_help --clips 10
python -u .\src\main.py build-dataset --version v1
python -u .\src\main.py train-seq --version v1 --model-name isl_intent_v1
python -u .\src\main.py evaluate --version v1 --model-name isl_intent_v1
python -u .\src\main.py promote --model-name isl_intent_v1
python -u .\src\main.py serve-api --port 8000
```

## Latency Target

The system prints rolling median end-to-end latency and stage timings while running.
Current goal: <=200 ms median on target hardware.
