param(
    [Parameter(Mandatory=$true)][string]$WorkDir,
    [string]$ArchivePath,
    [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'reports\OFFLINE_ENVIRONMENT.json') -Raw | ConvertFrom-Json
$workRoot = [System.IO.Path]::GetFullPath($WorkDir)
New-Item -ItemType Directory -Path $workRoot -Force | Out-Null
$bundleRoot = Join-Path $workRoot 'offline-environment'
$cachedArchive = Join-Path $workRoot $manifest.github.asset
if ($ArchivePath) {
    $cachedArchive = [System.IO.Path]::GetFullPath($ArchivePath)
} elseif (-not (Test-Path -LiteralPath $cachedArchive)) {
    $url = "https://github.com/$($manifest.github.repository)/releases/download/$($manifest.github.release)/$($manifest.github.asset)"
    Invoke-WebRequest -Uri $url -OutFile $cachedArchive
}
if ((Get-FileHash -LiteralPath $cachedArchive -Algorithm SHA256).Hash -ne $manifest.archive_sha256) {
    throw 'Offline environment archive checksum mismatch.'
}
if (-not (Test-Path -LiteralPath $bundleRoot)) {
    Expand-Archive -LiteralPath $cachedArchive -DestinationPath $bundleRoot
}
foreach ($record in $manifest.files) {
    $payload = [System.IO.Path]::GetFullPath((Join-Path $bundleRoot $record.path))
    if (-not $payload.StartsWith($bundleRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Payload path escapes the bundle: $($record.path)"
    }
    if ((Get-Item -LiteralPath $payload).Length -ne $record.size -or
        (Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash -ne $record.sha256) {
        throw "Offline environment payload checksum mismatch: $($record.path)"
    }
}
$requirements = Join-Path $bundleRoot 'requirements.lock'
if ((Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash -ne $manifest.requirements_sha256) {
    throw 'Bundled requirements checksum mismatch.'
}
$python = Join-Path $bundleRoot 'python\python.exe'
$venvRoot = Join-Path $workRoot '.venv'
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'
if (-not $SkipInstall) {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $python -m venv $venvRoot
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    & $venvPython -m pip install --no-index --find-links (Join-Path $bundleRoot 'wheels') --require-hashes -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Offline dependency installation failed.' }
}
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HOME = Join-Path $workRoot 'cache\huggingface'
$env:HF_MODULES_CACHE = Join-Path $workRoot 'cache\modules'
Write-Output "Environment ready. Python: $venvPython"
Write-Output "Run from the repository. PYTHONPATH is set for this PowerShell session."
