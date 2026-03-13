# SignifyAI

SignifyAI is an ISL-focused assistive communication system designed for continuous sign-to-speech translation.

## Current Implementation (Vertical Slice)

- Streaming runtime with stage timing probes (`capture`, `perception`, `decode`, `speech`, `render`)
- Baseline intent decoder for task-focused phrase pack
- Landmark-first recording workflow with quality reports
- Dataset version builder with signer-aware split manifests
- Sequence model training/evaluation/promote pipeline
- Model registry with active/promoted versions

## Quick Start

```bash
python -m pip install -r requirements.txt
python -u .\src\main.py -h
```

If your IDE still shows import errors (`numpy`, `mediapipe`, `sklearn`, `fastapi`, etc.):
- ensure your editor interpreter is the same one where you ran `pip install -r requirements.txt`
- restart the editor language server after installation

## CLI

```bash
python -u .\src\main.py run
python -u .\src\main.py record --intent hospital_help --clips 10
python -u .\src\main.py build-dataset --version v1
python -u .\src\main.py dataset-health --version v1
python -u .\src\main.py train-seq --version v1 --model-name signifyai_global
python -u .\src\main.py evaluate --version v1 --model-name signifyai_global
python -u .\src\main.py promote --model-name signifyai_global
```

The default production model id is `signifyai_global`.

Training guardrails:
- training is blocked if train split has fewer than 2 labels
- dataset health reports missing files, empty val/test splits, and signer coverage
- `train-seq` now benchmarks multiple ML models and saves the best single model automatically

## Latency Target

The system prints rolling median end-to-end latency and stage timings while running.
Current goal: <=200 ms median on target hardware.
