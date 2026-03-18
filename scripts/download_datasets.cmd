@echo off
setlocal
setlocal EnableDelayedExpansion

set DATASETS=datamunge/sign-language-mnist ardamavi/sign-language-digits-dataset

if not "%ISL_DATASET_ID%"=="" (
  set DATASETS=%DATASETS% %ISL_DATASET_ID%
) else (
  echo [note] ISL_DATASET_ID not set. Skipping ISL Kaggle download (you can still place it manually).
)

set ARGS=scripts/bootstrap_data.py --download-kaggle
for %%D in (%DATASETS%) do (
  set ARGS=!ARGS! --dataset %%D
)

python !ARGS!
endlocal
