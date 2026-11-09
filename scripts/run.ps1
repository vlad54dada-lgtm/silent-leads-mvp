# Повний прогін на Windows-машині розробника.
# Обходить обмеження Application Control (системний Python, python -m) і підхоплює ключі API
# зі змінних середовища користувача, якщо в поточній сесії їх немає.
# Ключі ніде не друкуються і нікуди, крім обраного в config.yaml провайдера, не передаються.
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

# Яка саме змінна потрібна, визначає ai_review.api_key_env у конфігу; тут підхоплюємо обидві відомі.
$found = $false
foreach ($name in @("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")) {
    $current = [Environment]::GetEnvironmentVariable($name, "Process")
    if ([string]::IsNullOrEmpty($current)) {
        $current = [Environment]::GetEnvironmentVariable($name, "User")
        if (-not [string]::IsNullOrEmpty($current)) {
            [Environment]::SetEnvironmentVariable($name, $current, "Process")
        }
    }
    if (-not [string]::IsNullOrEmpty($current)) { $found = $true }
}
if (-not $found) {
    Write-Warning "Ні ANTHROPIC_API_KEY, ні OPENROUTER_API_KEY не знайдено: AI-перевірка впаде в manual_review для всіх кандидатів."
}

uv run --no-sync python -m silent_leads.cli run --input $InputPath --config $ConfigPath --output $OutputDir
