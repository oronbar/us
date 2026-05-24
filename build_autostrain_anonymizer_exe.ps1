$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $repoRoot "autostrain_anonymizer.py"
$buildPath = Join-Path $repoRoot "build"
$distPath = Join-Path $repoRoot "dist"
$exeName = "AutoStrainCapAnonymizer"

if (-not (Test-Path $scriptPath)) {
    throw "Could not find $scriptPath"
}

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python -m PyInstaller --version *> $null
$pyInstallerExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousPreference
if ($pyInstallerExitCode -ne 0) {
    Write-Host "PyInstaller is not installed. Installing it for the current user..."
    python -m pip install --user pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller."
    }
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --workpath $buildPath `
    --specpath $buildPath `
    --distpath $distPath `
    --name $exeName `
    $scriptPath

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

$exePath = Join-Path $distPath "$exeName.exe"
if (-not (Test-Path $exePath)) {
    throw "Expected output was not created: $exePath"
}

Write-Host ""
Write-Host "Built: $exePath"
