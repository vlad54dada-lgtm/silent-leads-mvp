# Діагностика ключа AI-провайдера на Windows (ключ ніколи не друкується).
param([string]$ConfigPath = "config.yaml")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:UV_PYTHON_PREFERENCE = "only-system"
$env:PYTHONUTF8 = "1"
# Системний Python: з UV_PYTHON, якщо задано; інакше перший python.exe у PATH.
if ([string]::IsNullOrEmpty($env:UV_PYTHON)) {
    $py = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $py) {
        throw "Не знайдено python.exe у PATH. Задайте змінну UV_PYTHON зі шляхом до системного Python 3.12+."
    }
    $env:UV_PYTHON = $py.Source
}

foreach ($name in @("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")) {
    if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name, "Process"))) {
        $value = [Environment]::GetEnvironmentVariable($name, "User")
        if (-not [string]::IsNullOrEmpty($value)) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

uv run --no-sync python scripts/check_ai_key.py $ConfigPath
