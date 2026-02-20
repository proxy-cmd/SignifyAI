param([Parameter(Mandatory=$true)][string]$Script)
& .\.venv\Scripts\python.exe -u $Script
