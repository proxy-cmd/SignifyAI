# SignifyAI Research Roadmap

This file tracks practical upgrades to move from prototype-level gesture labels toward continuous sign-language understanding.

## Verified datasets to use next

1. WLASL (word-level ASL, large vocabulary)  
   https://arxiv.org/abs/1910.11006
2. How2Sign (continuous ASL, multimodal, 80+ hours)  
   https://how2sign.github.io/  
   https://arxiv.org/abs/2008.08143
3. RWTH-PHOENIX-Weather 2014/2014T (continuous CSLR/SLT benchmark)  
   https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

## Near-term implementation plan (inside this repo)

1. Add sequence dataset builder
- Save fixed-length landmark windows (e.g., 24-48 frames) with labels.
- Export `data/processed/sequence_dataset.npz`.

2. Add temporal model training
- Baseline: temporal classifier using frame-window features (flattened + MLP / tree model).
- Next: RNN/TCN/Transformer sequence model in a separate training script/notebook.

3. Add online latency-accuracy control
- Keep current adaptive FPS.
- Add wait-k style decision buffer for smoother real-time output.

4. Add signer adaptation mode
- 15-30 second personal calibration session.
- Per-signer threshold tuning saved to profile JSON.

5. Add non-manual cues (future)
- Add face/head cues when available to improve sentence-level meaning.

## Current status (already implemented)

1. Rule-based live demo with robust hand signs and speech.
2. AutoML training pipeline with confusion-matrix export.
3. One-command bootstrap: Kaggle import -> image landmarks -> AutoML.
4. Stage mode, demo script, recording, benchmark, doctor checks.

