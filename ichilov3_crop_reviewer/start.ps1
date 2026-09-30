$ErrorActionPreference = 'Stop'
$reviewProject = Split-Path -Parent $PSScriptRoot
$reviewOutput = 'D:\DS\ichilov3_crop_review'
New-Item -ItemType Directory -Path $reviewOutput -Force | Out-Null
$reviewListening = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8770 -State Listen -ErrorAction SilentlyContinue
if (-not $reviewListening) {
    $reviewProcess = Start-Process -FilePath (Join-Path $reviewProject '.venv\Scripts\python.exe') -ArgumentList @('-u', (Join-Path $PSScriptRoot 'server.py')) -WorkingDirectory $reviewProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $reviewOutput 'server.log') -RedirectStandardError (Join-Path $reviewOutput 'server_errors.log') -PassThru
    $reviewProcess.Id | Set-Content -LiteralPath (Join-Path $reviewOutput 'server.pid')
}
& 'C:\Program Files\Tailscale\tailscale.exe' serve --bg --http=8770 http://127.0.0.1:8770
Write-Output 'Review: http://oron-desktop.tailbe45ac.ts.net:8770/'
