# Діагностика ключа AI-провайдера на Windows-машині розробника (ключ ніколи не друкується).
param([string]$ConfigPath = "config.yaml")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:UV_PYTHON_PREFERENCE = "only-system"
$env:UV_PYTHON = "C:/Users/Acer/AppData/Local/Python/pythoncore-3.14-64/python.exe"
$env:PYTHONUTF8 = "1"

foreach ($name in @("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")) {
    if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name, "Process"))) {
        $value = [Environment]::GetEnvironmentVariable($name, "User")
        if (-not [string]::IsNullOrEmpty($value)) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

uv run --no-sync python scripts/check_ai_key.py $ConfigPath
