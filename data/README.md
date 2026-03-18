Data Folder Guide
=================

This project keeps large runtime and dataset files out of git by default.

What is tracked
---------------
- small placeholders
- taught-sign prototype file path
- minimal folder structure

What is NOT tracked
-------------------
- large external datasets
- model binaries
- generated landmarks and logs
- exports

Important runtime paths
-----------------------
- Taught sign prototypes: data/models/sign_prototypes.json
- Model registry: data/models/registry.json
- Hand detector model: data/models/hand_landmarker.task
- Face detector model: data/models/face_landmarker.task
- Live taught clips: data/landmarks/raw/live_teach/
- External datasets: data/external/

Tip
---
Use scripts/bootstrap_data.py to create required local folders/placeholders and print missing runtime files.

For tester handoff, share a zip that already contains the required `data/models` and `data/landmarks` content.
