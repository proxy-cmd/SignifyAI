# ML Checkpoint (Safe Snapshot)

Date: 2026-03-13

Status:
- Kaggle API auth configured on this machine (`KAGGLE_API_TOKEN` in user env)
- Dataset used: `datamunge/sign-language-mnist` (Kaggle)
- Current stable benchmark target: classical ML baseline

Checkpoint result:
- Model: `ExtraTreesClassifier(n_estimators=500)`
- Test accuracy: `82.68%`
- Train samples: `27,455`
- Test samples: `7,172`
- Runtime (train + eval): `~63s`

Notes:
- This is the safe baseline checkpoint to continue from tomorrow.
- Goal path is unchanged: improve from ~82% -> 90% -> 95% -> 98%.
