Param(
    [string]$ISLDatasetId = $env:ISL_DATASET_ID
)

$ErrorActionPreference = "Stop"

$datasets = @(
    "datamunge/sign-language-mnist",
    "ardamavi/sign-language-digits-dataset"
)

if ($ISLDatasetId -and $ISLDatasetId.Trim().Length -gt 0) {
    $datasets += $ISLDatasetId.Trim()
} else {
    Write-Host "[note] ISL_DATASET_ID not set. Skipping ISL Kaggle download (you can still place it manually)."
}

$args = @("scripts/bootstrap_data.py", "--download-kaggle")
foreach ($d in $datasets) {
    $args += @("--dataset", $d)
}

python @args
