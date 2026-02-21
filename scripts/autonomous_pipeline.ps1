param(
    [switch]$SkipBootstrap = $false,
    [string]$KaggleSlug = "grassknoted/asl-alphabet",
    [int]$MaxPerClass = 800
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "data\processed\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "autonomous_$stamp.log"

function Run-Step($name, $cmd) {
    Write-Host "==== $name ====" -ForegroundColor Cyan
    Add-Content -Path $logFile -Value "`n==== $name ====`n"
    Write-Host $cmd
    Add-Content -Path $logFile -Value $cmd
    cmd /c $cmd 2>&1 | Tee-Object -FilePath $logFile -Append
}

Run-Step "Doctor checks" "python -u .\src\main.py doctor"
Run-Step "Benchmark" "python -u .\src\main.py benchmark --seconds 5"

if (-not $SkipBootstrap) {
    Run-Step "Bootstrap ML pipeline" "python -u .\src\main.py bootstrap-ml --slug $KaggleSlug --max-per-class $MaxPerClass"
} else {
    Write-Host "Skipping bootstrap training." -ForegroundColor Yellow
}

Run-Step "Generate report" "python -u .\src\main.py report"
Write-Host "Autonomous pipeline completed. Log: $logFile" -ForegroundColor Green

