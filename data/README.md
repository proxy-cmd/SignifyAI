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
- large Kaggle datasets
- model binaries
- generated landmarks and logs
- exports

Important runtime paths
-----------------------
- Taught sign prototypes: data/models/sign_prototypes.json
- Live taught clips: data/landmarks/raw/live_teach/
- External datasets: data/external/kaggle/

Tip
---
Use scripts/bootstrap_data.py (or wrapper scripts in /scripts) to create required folders and optional dataset downloads.
