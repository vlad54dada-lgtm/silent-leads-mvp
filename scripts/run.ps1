# Повний прогін на Windows (обхід Application Control: системний Python, python -m).
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
$env:PYTHONUTF8 = "1"
# Системний Python: з UV_PYTHON, якщо задано; інакше перший python.exe у PATH.
if ([string]::IsNullOrEmpty($env:UV_PYTHON)) {
    $py = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $py) {
        throw "Не знайдено python.exe у PATH. Задайте змінну UV_PYTHON зі шляхом до системного Python 3.12+."
    }
    $env:UV_PYTHON = $py.Source
}

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

# Явно показуємо, чи підуть тексти діалогів до зовнішньої моделі, і куди саме.
$aiOn = Select-String -Path $ConfigPath -Pattern '^\s*enabled:\s*true' -Quiet
if ($aiOn) {
    $gw = Select-String -Path $ConfigPath -Pattern '^\s*base_url:\s*(\S+)' | Select-Object -First 1
    $where = if ($null -ne $gw -and $gw.Matches[0].Groups[1].Value -ne "null") { $gw.Matches[0].Groups[1].Value } else { "api.anthropic.com" }
    Write-Warning "AI-аналіз УВІМКНЕНО в ${ConfigPath}: діалоги кейсів буде надіслано до $where. Ctrl+C протягом 5 с, щоб скасувати."
    Start-Sleep -Seconds 5
} else {
    Write-Host "AI-аналіз вимкнено ($ConfigPath): прогін повністю офлайн."
}

uv run --no-sync python -m silent_leads.cli run --input $InputPath --config $ConfigPath --output $OutputDir
