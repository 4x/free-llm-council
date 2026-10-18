<#
.SYNOPSIS
    Launches the llm-council interactive web interface and opens the default browser.
.DESCRIPTION
    Changes to the repository root so .env and council.duckdb resolve regardless of
    where the script or desktop shortcut is started from, then runs the web server
    via uv and opens the default browser once the server is bound.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    $Host.UI.RawUI.WindowTitle = "llm-council - Web UI"
} catch {}

Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv was not found on PATH. Install uv (https://docs.astral.sh/uv/) and try again."
    Read-Host "Press Enter to exit"
    exit 1
}

& uv run llm-council --web --open-browser @args
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0 -and $exitCode -ne 130) {
    Write-Host ""
    Write-Host "llm-council exited with status $exitCode." -ForegroundColor Red
    Read-Host "Press Enter to close this window"
}
exit $exitCode
