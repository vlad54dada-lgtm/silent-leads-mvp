# Повний прогін на Windows-машині розробника.
# Обходить обмеження Application Control (системний Python, python -m) і підхоплює
# ANTHROPIC_API_KEY зі змінних середовища користувача, якщо в поточній сесії його немає.
# Ключ ніде не друкується і нікуди, крім Anthropic API, не передається.
param(
    [string]$InputPath = "data/input/export.xlsx",
    [string]$ConfigPath = "config.yaml",
    [string]$OutputDir = "data/output"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:UV_PYTHON_PREFERENCE = "only-system"
$env:UV_PYTHON = "C:/Users/Acer/AppData/Local/Python/pythoncore-3.14-64/python.exe"
$env:PYTHONUTF8 = "1"

if ([string]::IsNullOrEmpty($env:ANTHROPIC_API_KEY)) {
    $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")
}
if ([string]::IsNullOrEmpty($env:ANTHROPIC_API_KEY)) {
    Write-Warning "ANTHROPIC_API_KEY не знайдено: AI-перевірка впаде в manual_review для всіх кандидатів."
}

uv run --no-sync python -m silent_leads.cli run --input $InputPath --config $ConfigPath --output $OutputDir
