[CmdletBinding()]
param(
    [ValidateSet('Dashboard', 'Start', 'Resume', 'Pause', 'Stop', 'Status', 'Worker')]
    [string]$Mode = 'Dashboard'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:Version = '1.0.1'
$script:ToolName = 'SZMC2 Google Drive Batch Sync'
$script:ScriptPath = $MyInvocation.MyCommand.Path
$script:ToolDirectory = Split-Path -Parent $script:ScriptPath
$script:ConfigPath = Join-Path $script:ToolDirectory 'config.json'
$script:StateDirectory = Join-Path $env:LOCALAPPDATA 'SZMC2GoogleDriveBatchSync'
$script:ControlPath = Join-Path $script:StateDirectory 'control.json'
$script:StatusPath = Join-Path $script:StateDirectory 'status.json'
$script:LogPath = Join-Path $script:StateDirectory 'sync.log'
$script:RobocopyLogPath = Join-Path $script:StateDirectory 'robocopy.log'
$script:WorkerPidPath = Join-Path $script:StateDirectory 'worker.pid'

function Initialize-StateDirectory {
    if (-not (Test-Path -LiteralPath $script:StateDirectory)) {
        New-Item -ItemType Directory -Path $script:StateDirectory -Force | Out-Null
    }
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            # Windows PowerShell 5.1 treats UTF-8 files without a BOM as ANSI.
            # Read explicitly as UTF-8 so localized Google Drive mount names work.
            $json = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
            return ($json | ConvertFrom-Json)
        }
        catch { Start-Sleep -Milliseconds (100 * $attempt) }
    }
    return $null
}

function Write-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)]$Value)
    $json = $Value | ConvertTo-Json -Depth 8
    $lastError = $null
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        $temporaryPath = '{0}.{1}.{2}.tmp' -f $Path, $PID, ([guid]::NewGuid().ToString('N'))
        try {
            [System.IO.File]::WriteAllText($temporaryPath, $json, (New-Object System.Text.UTF8Encoding($false)))
            [System.IO.File]::Copy($temporaryPath, $Path, $true)
            Remove-Item -LiteralPath $temporaryPath -Force
            return
        }
        catch {
            $lastError = $_
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds (150 * $attempt)
        }
    }
    throw $lastError
}

function Get-Config {
    $config = Read-JsonFile -Path $script:ConfigPath
    if ($null -eq $config) { throw "Cannot read configuration: $script:ConfigPath" }
    return $config
}

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    Initialize-StateDirectory
    Add-Content -LiteralPath $script:LogPath -Value ('{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message) -Encoding UTF8
}

function Get-ControlState {
    $control = Read-JsonFile -Path $script:ControlPath
    if ($null -eq $control -or -not $control.PSObject.Properties['desiredState']) { return 'stopped' }
    return [string]$control.desiredState
}

function Set-ControlState {
    param([ValidateSet('running', 'paused', 'stopped')][string]$DesiredState)
    Initialize-StateDirectory
    Write-JsonFile -Path $script:ControlPath -Value ([ordered]@{
        desiredState = $DesiredState
        changedAt = (Get-Date).ToString('o')
        changedByPid = $PID
    })
}

function Get-WorkerPid {
    if (-not (Test-Path -LiteralPath $script:WorkerPidPath)) { return $null }
    try {
        $workerPid = [int](Get-Content -LiteralPath $script:WorkerPidPath -Raw)
        $process = Get-Process -Id $workerPid -ErrorAction Stop
        if ($process.ProcessName -notlike '*powershell*' -and $process.ProcessName -notlike '*pwsh*') { return $null }
        return $workerPid
    }
    catch { return $null }
}

function Start-WorkerProcess {
    Initialize-StateDirectory
    $existingPid = Get-WorkerPid
    if ($null -ne $existingPid) { return $existingPid }
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $script:ScriptPath), '-Mode', 'Worker')
    $process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru
    return $process.Id
}

function Get-FreeSpaceGB {
    param([Parameter(Mandatory = $true)][string]$Path)
    $drive = New-Object System.IO.DriveInfo([System.IO.Path]::GetPathRoot($Path))
    return [math]::Round($drive.AvailableFreeSpace / 1GB, 2)
}

function Test-GoogleDriveRunning {
    return ($null -ne (Get-Process -Name 'GoogleDriveFS' -ErrorAction SilentlyContinue | Select-Object -First 1))
}

function Get-FolderInfo {
    param([Parameter(Mandatory = $true)][string]$Path)
    $count = 0L
    $bytes = 0L
    if (Test-Path -LiteralPath $Path) {
        foreach ($file in Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction Stop) {
            $count++
            $bytes += $file.Length
        }
    }
    return [pscustomobject]@{ FileCount = $count; Bytes = $bytes }
}

function Test-FolderMatches {
    param([Parameter(Mandatory = $true)][string]$SourcePath, [Parameter(Mandatory = $true)][string]$DestinationPath)
    if (-not (Test-Path -LiteralPath $DestinationPath -PathType Container)) { return $false }
    $sourceInfo = Get-FolderInfo -Path $SourcePath
    $destinationInfo = Get-FolderInfo -Path $DestinationPath
    return ($sourceInfo.FileCount -eq $destinationInfo.FileCount -and $sourceInfo.Bytes -eq $destinationInfo.Bytes)
}

function Update-Status {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$CurrentFolder = '',
        [int]$CompletedFolders = 0,
        [int]$TotalFolders = 0,
        [double]$CompletedGB = 0,
        [double]$TotalGB = 0
    )
    $config = Get-Config
    $status = [ordered]@{
        toolVersion = $script:Version
        workerPid = $PID
        phase = $Phase
        desiredState = Get-ControlState
        message = $Message
        currentFolder = $CurrentFolder
        completedFolders = $CompletedFolders
        totalFolders = $TotalFolders
        completedGB = [math]::Round($CompletedGB, 2)
        totalGB = [math]::Round($TotalGB, 2)
        cFreeGB = Get-FreeSpaceGB -Path ([string]$config.LocalCacheDrive)
        reserveFreeGB = [double]$config.ReserveFreeGB
        updatedAt = (Get-Date).ToString('o')
        destinationPath = [string]$config.DestinationPath
        logPath = $script:LogPath
    }
    Write-JsonFile -Path $script:StatusPath -Value $status
}

function Wait-ForRunPermission {
    param([int]$CompletedFolders, [int]$TotalFolders, [double]$CompletedGB, [double]$TotalGB, [string]$CurrentFolder = '')
    while ($true) {
        $desired = Get-ControlState
        if ($desired -eq 'stopped') { return $false }
        if ($desired -eq 'running') { return $true }
        Update-Status -Phase 'Paused' -Message 'Paused. Google Drive may continue uploading data already staged in its cache.' -CurrentFolder $CurrentFolder -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
        Start-Sleep -Seconds 2
    }
}

function Wait-ForCapacityAndDrive {
    param([Parameter(Mandatory = $true)]$Config, [double]$FolderGB, [int]$CompletedFolders, [int]$TotalFolders, [double]$CompletedGB, [double]$TotalGB, [string]$CurrentFolder)
    while ($true) {
        if (-not (Wait-ForRunPermission -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB -CurrentFolder $CurrentFolder)) { return $false }
        if (-not (Test-GoogleDriveRunning) -or -not (Test-Path -LiteralPath ([string]$Config.GoogleDriveRoot))) {
            Update-Status -Phase 'WaitingForGoogleDrive' -Message 'Google Drive for desktop is not available. Start it; the worker will keep waiting.' -CurrentFolder $CurrentFolder -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
            Start-Sleep -Seconds 10
            continue
        }
        $freeGB = Get-FreeSpaceGB -Path ([string]$Config.LocalCacheDrive)
        $requiredGB = [double]$Config.ReserveFreeGB + [math]::Min($FolderGB, [double]$Config.MaxExpectedCacheGB)
        if ($freeGB -lt $requiredGB) {
            Update-Status -Phase 'WaitingForCacheSpace' -Message ('Waiting for Google Drive to drain its cache. C: free {0:N1} GB; need {1:N1} GB before continuing.' -f $freeGB, $requiredGB) -CurrentFolder $CurrentFolder -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
            Start-Sleep -Seconds 20
            continue
        }
        return $true
    }
}

function Invoke-ThrottledRobocopy {
    param([Parameter(Mandatory = $true)]$Item, [Parameter(Mandatory = $true)]$Config, [int]$CompletedFolders, [int]$TotalFolders, [double]$CompletedGB, [double]$TotalGB)
    $arguments = @(
        ('"{0}"' -f $Item.SourcePath), ('"{0}"' -f $Item.DestinationPath),
        '/E', '/Z', '/J', ('/IPG:{0}' -f [int]$Config.InterPacketGapMs),
        '/COPY:DAT', '/DCOPY:DAT', '/R:3', '/W:10', '/XJ',
        '/NP', '/NFL', '/NDL', ('/LOG+:{0}' -f $script:RobocopyLogPath)
    )
    while ($true) {
        if (-not (Wait-ForCapacityAndDrive -Config $Config -FolderGB ([double]$Item.GB) -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB -CurrentFolder $Item.Name)) { return 'stopped' }
        Update-Status -Phase 'Copying' -Message ('Copying/resuming {0} to the Google Drive streamed mount.' -f $Item.Name) -CurrentFolder $Item.Name -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
        Write-Log ('Starting throttled robocopy for {0}' -f $Item.Name)
        $process = Start-Process -FilePath 'robocopy.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru
        $interruptedForSpace = $false
        while (-not $process.HasExited) {
            Start-Sleep -Seconds 2
            $process.Refresh()
            $desired = Get-ControlState
            $freeGB = Get-FreeSpaceGB -Path ([string]$Config.LocalCacheDrive)
            if ($desired -eq 'paused' -or $desired -eq 'stopped' -or $freeGB -lt [double]$Config.ReserveFreeGB) {
                $reason = if ($freeGB -lt [double]$Config.ReserveFreeGB) { 'cache reserve reached' } else { $desired }
                Write-Log ('Interrupting robocopy for {0}: {1}' -f $Item.Name, $reason)
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                $process.WaitForExit()
                if ($desired -eq 'stopped') { return 'stopped' }
                if ($freeGB -lt [double]$Config.ReserveFreeGB) { $interruptedForSpace = $true }
                break
            }
        }
        if ($interruptedForSpace) { continue }
        if ((Get-ControlState) -eq 'paused') { continue }
        $exitCode = $process.ExitCode
        if ($exitCode -le 7) {
            if (Test-FolderMatches -SourcePath $Item.SourcePath -DestinationPath $Item.DestinationPath) {
                Write-Log ('Verified destination inventory for {0}; robocopy code {1}' -f $Item.Name, $exitCode)
                return 'copied'
            }
            Write-Log ('Inventory mismatch after robocopy for {0}; retrying' -f $Item.Name)
        }
        else {
            Write-Log ('Robocopy failed for {0} with code {1}; retrying' -f $Item.Name, $exitCode)
        }
        Update-Status -Phase 'Retrying' -Message ('Retrying {0}; see robocopy.log for details.' -f $Item.Name) -CurrentFolder $Item.Name -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
        Start-Sleep -Seconds 15
    }
}

function Invoke-Worker {
    Initialize-StateDirectory
    $mutex = New-Object System.Threading.Mutex($false, 'Local\OneDriveDatasetBatchSyncGlobalWorker')
    $hasMutex = $false
    try {
        Set-Content -LiteralPath $script:WorkerPidPath -Value $PID -Encoding ASCII
        $config = Get-Config
        while (-not $hasMutex) {
            $hasMutex = $mutex.WaitOne(0, $false)
            if ($hasMutex) { break }
            if ((Get-ControlState) -eq 'stopped') { return }
            Update-Status -Phase 'WaitingForOtherDataset' -Message 'Waiting for the OneDrive dataset worker to stop and release the shared C: cache lock.'
            Start-Sleep -Seconds 5
        }
        if (-not (Test-Path -LiteralPath ([string]$config.SourcePath))) { throw "Source is unavailable: $($config.SourcePath)" }
        if (-not (Test-GoogleDriveRunning)) { throw 'Google Drive for desktop is not running.' }
        if (-not (Test-Path -LiteralPath ([string]$config.GoogleDriveRoot))) { throw "Google Drive mount is unavailable: $($config.GoogleDriveRoot)" }
        if (-not (Test-Path -LiteralPath ([string]$config.DestinationPath))) {
            New-Item -ItemType Directory -Path ([string]$config.DestinationPath) -Force | Out-Null
        }

        Write-Log ('Worker started (PID {0})' -f $PID)
        Update-Status -Phase 'Scanning' -Message 'Scanning source folders and resumable Google Drive destination.'
        $inventory = @()
        $totalBytes = 0L
        foreach ($folder in Get-ChildItem -LiteralPath ([string]$config.SourcePath) -Directory -Force | Sort-Object Name) {
            $info = Get-FolderInfo -Path $folder.FullName
            $totalBytes += $info.Bytes
            $inventory += [pscustomobject]@{
                Name = $folder.Name
                SourcePath = $folder.FullName
                DestinationPath = Join-Path ([string]$config.DestinationPath) $folder.Name
                Bytes = $info.Bytes
                GB = [math]::Round($info.Bytes / 1GB, 4)
            }
        }
        $totalFolders = $inventory.Count
        $totalGB = $totalBytes / 1GB
        $completed = @()
        $pending = @()
        $completedBytes = 0L
        foreach ($item in $inventory) {
            if (Test-FolderMatches -SourcePath $item.SourcePath -DestinationPath $item.DestinationPath) {
                $completed += $item
                $completedBytes += $item.Bytes
            }
            else { $pending += $item }
        }
        $completedFolders = $completed.Count
        $completedGB = $completedBytes / 1GB
        Update-Status -Phase 'Ready' -Message ('Resume scan complete: {0} matching folders, {1} remaining.' -f $completedFolders, $pending.Count) -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB

        foreach ($item in $pending) {
            if (-not (Wait-ForRunPermission -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB -CurrentFolder $item.Name)) {
                Update-Status -Phase 'Stopped' -Message 'Stopped safely. Partial files will resume next time.' -CurrentFolder $item.Name -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                return
            }
            $result = Invoke-ThrottledRobocopy -Item $item -Config $config -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
            if ($result -eq 'stopped') {
                Update-Status -Phase 'Stopped' -Message 'Stopped safely. Partial files will resume next time.' -CurrentFolder $item.Name -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                return
            }
            $completedFolders++
            $completedGB += [double]$item.GB
            Update-Status -Phase 'FolderVerified' -Message ('Verified {0} in the Google Drive mount. Allowing cache/upload drain.' -f $item.Name) -CurrentFolder $item.Name -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
            Start-Sleep -Seconds ([int]$config.PostFolderDrainSeconds)
        }

        Set-ControlState -DesiredState 'stopped'
        Update-Status -Phase 'DriveMountComplete' -Message 'All folders match the Google Drive mount. Keep Google Drive running until its tray menu says Up to date.' -CompletedFolders $totalFolders -TotalFolders $totalFolders -CompletedGB $totalGB -TotalGB $totalGB
        Write-Log ('ALL {0} folders verified in Google Drive mount; waiting for Drive for desktop final background drain' -f $totalFolders)
    }
    catch {
        try {
            Write-Log ('FATAL [{0}]: {1}' -f $_.Exception.GetType().FullName, $_.Exception.Message)
            if ($_.ScriptStackTrace) { Write-Log ('STACK: {0}' -f ($_.ScriptStackTrace -replace "`r?`n", ' | ')) }
            $prior = Read-JsonFile -Path $script:StatusPath
            $cf = 0; $tf = 0; $cg = 0; $tg = 0; $current = ''
            if ($null -ne $prior) { $cf=[int]$prior.completedFolders; $tf=[int]$prior.totalFolders; $cg=[double]$prior.completedGB; $tg=[double]$prior.totalGB; $current=[string]$prior.currentFolder }
            Update-Status -Phase 'Error' -Message $_.Exception.Message -CurrentFolder $current -CompletedFolders $cf -TotalFolders $tf -CompletedGB $cg -TotalGB $tg
        }
        catch { }
    }
    finally {
        Remove-Item -LiteralPath $script:WorkerPidPath -Force -ErrorAction SilentlyContinue
        if ($hasMutex) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}

function Show-Dashboard {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()
    $form = New-Object System.Windows.Forms.Form
    $form.Text = $script:ToolName
    $form.Size = New-Object System.Drawing.Size(780, 535)
    $form.MinimumSize = New-Object System.Drawing.Size(720, 500)
    $form.StartPosition = 'CenterScreen'
    $form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

    $title = New-Object System.Windows.Forms.Label
    $title.Text = 'SZMC2 -> Google Drive'
    $title.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 16)
    $title.AutoSize = $true
    $title.Location = New-Object System.Drawing.Point(20, 18)
    $form.Controls.Add($title)

    $phase = New-Object System.Windows.Forms.Label
    $phase.Text = 'State: Not started'
    $phase.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 11)
    $phase.AutoSize = $true
    $phase.Location = New-Object System.Drawing.Point(22, 62)
    $form.Controls.Add($phase)

    $message = New-Object System.Windows.Forms.Label
    $message.Text = 'Use Start / Resume to begin.'
    $message.Location = New-Object System.Drawing.Point(22, 92)
    $message.Size = New-Object System.Drawing.Size(710, 45)
    $form.Controls.Add($message)

    $progress = New-Object System.Windows.Forms.ProgressBar
    $progress.Location = New-Object System.Drawing.Point(25, 143)
    $progress.Size = New-Object System.Drawing.Size(710, 25)
    $progress.Minimum = 0
    $progress.Maximum = 1
    $form.Controls.Add($progress)

    $details = New-Object System.Windows.Forms.Label
    $details.Text = 'Progress information will appear here.'
    $details.Location = New-Object System.Drawing.Point(22, 178)
    $details.Size = New-Object System.Drawing.Size(710, 48)
    $form.Controls.Add($details)

    $start = New-Object System.Windows.Forms.Button
    $start.Text = 'Start / Resume'; $start.Location = New-Object System.Drawing.Point(25,235); $start.Size = New-Object System.Drawing.Size(150,40); $form.Controls.Add($start)
    $pause = New-Object System.Windows.Forms.Button
    $pause.Text = 'Pause'; $pause.Location = New-Object System.Drawing.Point(190,235); $pause.Size = New-Object System.Drawing.Size(120,40); $form.Controls.Add($pause)
    $stop = New-Object System.Windows.Forms.Button
    $stop.Text = 'Stop'; $stop.Location = New-Object System.Drawing.Point(325,235); $stop.Size = New-Object System.Drawing.Size(120,40); $form.Controls.Add($stop)
    $openLog = New-Object System.Windows.Forms.Button
    $openLog.Text = 'Open log folder'; $openLog.Location = New-Object System.Drawing.Point(460,235); $openLog.Size = New-Object System.Drawing.Size(130,40); $form.Controls.Add($openLog)
    $openDestination = New-Object System.Windows.Forms.Button
    $openDestination.Text = 'Open destination'; $openDestination.Location = New-Object System.Drawing.Point(600,235); $openDestination.Size = New-Object System.Drawing.Size(135,40); $form.Controls.Add($openDestination)

    $logBox = New-Object System.Windows.Forms.TextBox
    $logBox.Location = New-Object System.Drawing.Point(25,295); $logBox.Size = New-Object System.Drawing.Size(710,155); $logBox.Multiline=$true; $logBox.ReadOnly=$true; $logBox.ScrollBars='Vertical'; $logBox.Font=New-Object System.Drawing.Font('Consolas',8.5); $form.Controls.Add($logBox)
    $note = New-Object System.Windows.Forms.Label
    $note.Text = 'Closing this window does not stop the worker. Use Stop before switching dataset tools.'; $note.Location=New-Object System.Drawing.Point(22,462); $note.AutoSize=$true; $form.Controls.Add($note)

    $refresh = {
        try {
            $s = Read-JsonFile -Path $script:StatusPath
            if ($null -ne $s) {
                $phase.Text = 'State: {0}' -f $s.phase
                $message.Text = [string]$s.message
                $max = [math]::Max(1,[int]$s.totalFolders); $progress.Maximum=$max; $progress.Value=[math]::Min($max,[int]$s.completedFolders)
                $details.Text = ('Folders: {0}/{1}    Data: {2:N1}/{3:N1} GB    Free C:: {4:N1} GB (reserve {5:N0})`r`nCurrent: {6}' -f [int]$s.completedFolders,[int]$s.totalFolders,[double]$s.completedGB,[double]$s.totalGB,[double]$s.cFreeGB,[double]$s.reserveFreeGB,[string]$s.currentFolder)
            }
            if (Test-Path -LiteralPath $script:LogPath) { $logBox.Lines=@(Get-Content -LiteralPath $script:LogPath -Tail 12 -ErrorAction SilentlyContinue); $logBox.SelectionStart=$logBox.TextLength; $logBox.ScrollToCaret() }
        }
        catch { }
    }
    $start.Add_Click({ Set-ControlState -DesiredState 'running'; Start-WorkerProcess|Out-Null; Write-Log 'Start/Resume requested from dashboard'; & $refresh })
    $pause.Add_Click({ Set-ControlState -DesiredState 'paused'; Write-Log 'Pause requested from dashboard'; & $refresh })
    $stop.Add_Click({ Set-ControlState -DesiredState 'stopped'; Write-Log 'Stop requested from dashboard'; & $refresh })
    $openLog.Add_Click({ Initialize-StateDirectory; Start-Process explorer.exe -ArgumentList ('"{0}"' -f $script:StateDirectory) })
    $openDestination.Add_Click({ Start-Process explorer.exe -ArgumentList ('"{0}"' -f [string](Get-Config).DestinationPath) })
    $timer=New-Object System.Windows.Forms.Timer; $timer.Interval=2000; $timer.Add_Tick($refresh); $timer.Start(); & $refresh; [void]$form.ShowDialog(); $timer.Stop()
}

Initialize-StateDirectory
switch ($Mode) {
    'Dashboard' { Show-Dashboard }
    'Start' { Set-ControlState -DesiredState 'running'; $workerPid=Start-WorkerProcess; Write-Output "Started/resumed worker PID $workerPid" }
    'Resume' { Set-ControlState -DesiredState 'running'; $workerPid=Start-WorkerProcess; Write-Output "Started/resumed worker PID $workerPid" }
    'Pause' { Set-ControlState -DesiredState 'paused'; Write-Output 'Pause requested' }
    'Stop' { Set-ControlState -DesiredState 'stopped'; Write-Output 'Stop requested' }
    'Status' { $status=Read-JsonFile -Path $script:StatusPath; if($null -eq $status){Write-Output 'Not started'}else{$status|ConvertTo-Json -Depth 8} }
    'Worker' { Invoke-Worker }
}
