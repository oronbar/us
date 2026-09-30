[CmdletBinding()]
param(
    [ValidateSet('Dashboard', 'Start', 'Resume', 'Pause', 'Stop', 'Status', 'Worker')]
    [string]$Mode = 'Dashboard',
    [string]$ConfigurationPath = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:ToolVersion = '1.1.3'
$script:ScriptPath = $MyInvocation.MyCommand.Path
$script:ToolDirectory = Split-Path -Parent $script:ScriptPath
if ([string]::IsNullOrWhiteSpace($ConfigurationPath)) {
    $script:ConfigPath = Join-Path $script:ToolDirectory 'config.json'
}
else {
    $script:ConfigPath = [System.IO.Path]::GetFullPath($ConfigurationPath)
}
$bootstrapConfig = Get-Content -LiteralPath $script:ConfigPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
$stateKey = if ($bootstrapConfig.PSObject.Properties['StateKey']) { [string]$bootstrapConfig.StateKey } else { 'SZMC2OneDriveBatchSync' }
$script:ToolName = if ($bootstrapConfig.PSObject.Properties['ToolName']) { [string]$bootstrapConfig.ToolName } else { 'SZMC2 OneDrive Batch Sync' }
$script:DatasetLabel = if ($bootstrapConfig.PSObject.Properties['DatasetLabel']) { [string]$bootstrapConfig.DatasetLabel } else { 'SZMC2' }
$script:StateDirectory = Join-Path $env:LOCALAPPDATA $stateKey
$script:ControlPath = Join-Path $script:StateDirectory 'control.json'
$script:StatusPath = Join-Path $script:StateDirectory 'status.json'
$script:LogPath = Join-Path $script:StateDirectory 'sync.log'
$script:WorkerPidPath = Join-Path $script:StateDirectory 'worker.pid'
$script:CloudRecallOnOpen = 0x00040000
$script:CloudPinned = 0x00080000
$script:CloudUnpinned = 0x00100000
$script:CloudRecallOnDataAccess = 0x00400000

function Initialize-StateDirectory {
    if (-not (Test-Path -LiteralPath $script:StateDirectory)) {
        New-Item -ItemType Directory -Path $script:StateDirectory -Force | Out-Null
    }
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try {
            return (Get-Content -LiteralPath $Path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop)
        }
        catch {
            Start-Sleep -Milliseconds 100
        }
    }
    return $null
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    $json = $Value | ConvertTo-Json -Depth 8
    $lastError = $null
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        $temporaryPath = '{0}.{1}.{2}.tmp' -f $Path, $PID, ([guid]::NewGuid().ToString('N'))
        try {
            [System.IO.File]::WriteAllText($temporaryPath, $json, (New-Object System.Text.UTF8Encoding($false)))
            # File.Replace with a null backup path is incompatible with some
            # Windows PowerShell/.NET Framework builds. Copy(overwrite) is
            # supported consistently; readers already retry partial reads.
            [System.IO.File]::Copy($temporaryPath, $Path, $true)
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction Stop
            return
        }
        catch {
            $lastError = $_
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds (100 * $attempt)
        }
    }
    throw $lastError
}

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    Initialize-StateDirectory
    $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
}

function Get-Config {
    $config = Read-JsonFile -Path $script:ConfigPath
    if ($null -eq $config) {
        throw "Cannot read configuration: $script:ConfigPath"
    }
    return $config
}

function Get-ControlState {
    $control = Read-JsonFile -Path $script:ControlPath
    if ($null -eq $control -or -not $control.PSObject.Properties['desiredState']) {
        return 'stopped'
    }
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
        if ($process.ProcessName -notlike '*powershell*' -and $process.ProcessName -notlike '*pwsh*') {
            return $null
        }
        return $workerPid
    }
    catch {
        return $null
    }
}

function Start-WorkerProcess {
    Initialize-StateDirectory
    $existingPid = Get-WorkerPid
    if ($null -ne $existingPid) { return $existingPid }

    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"{0}"' -f $script:ScriptPath),
        '-Mode', 'Worker',
        '-ConfigurationPath', ('"{0}"' -f $script:ConfigPath)
    )
    $process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru
    return $process.Id
}

function Get-FreeSpaceGB {
    param([Parameter(Mandatory = $true)][string]$Path)
    $root = [System.IO.Path]::GetPathRoot($Path)
    $drive = New-Object System.IO.DriveInfo($root)
    return [math]::Round($drive.AvailableFreeSpace / 1GB, 2)
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

function Test-IsCloudOnlyFile {
    param([Parameter(Mandatory = $true)][System.IO.FileInfo]$File)
    $attributes = [int]$File.Attributes
    $isUnpinned = (($attributes -band $script:CloudUnpinned) -ne 0)
    $hasRecallFlag = (($attributes -band $script:CloudRecallOnDataAccess) -ne 0) -or
                     (($attributes -band $script:CloudRecallOnOpen) -ne 0)
    return ($isUnpinned -and $hasRecallFlag)
}

function Get-CloudFolderState {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )
    if (-not (Test-Path -LiteralPath $DestinationPath)) {
        return [pscustomobject]@{ Ready = $false; SourceFiles = 0L; DestinationFiles = 0L; Bytes = 0L; LocalFiles = 0L }
    }

    $sourceInfo = Get-FolderInfo -Path $SourcePath
    $destinationCount = 0L
    $destinationBytes = 0L
    $localFiles = 0L
    foreach ($file in Get-ChildItem -LiteralPath $DestinationPath -File -Recurse -Force -ErrorAction Stop) {
        $destinationCount++
        $destinationBytes += $file.Length
        if (-not (Test-IsCloudOnlyFile -File $file)) { $localFiles++ }
    }
    $matchingInventory = ($sourceInfo.FileCount -eq $destinationCount) -and ($sourceInfo.Bytes -eq $destinationBytes)
    return [pscustomobject]@{
        Ready = ($matchingInventory -and $localFiles -eq 0)
        SourceFiles = $sourceInfo.FileCount
        DestinationFiles = $destinationCount
        Bytes = $sourceInfo.Bytes
        LocalFiles = $localFiles
    }
}

function Get-WorkInventory {
    param([Parameter(Mandatory = $true)]$Config)
    $sourceRoot = [string]$Config.SourcePath
    $destinationRoot = [string]$Config.DestinationPath
    $inventory = @()
    $mode = if ($Config.PSObject.Properties['InventoryMode']) { [string]$Config.InventoryMode } else { 'TopLevelFolders' }

    if ($mode -eq 'PatientFoldersWithUS') {
        foreach ($topLevelName in @($Config.IncludedTopLevelDirectories)) {
            $topLevelPath = Join-Path $sourceRoot ([string]$topLevelName)
            if (-not (Test-Path -LiteralPath $topLevelPath -PathType Container)) {
                throw "Configured source folder is missing: $topLevelPath"
            }
            foreach ($folder in Get-ChildItem -LiteralPath $topLevelPath -Directory -Recurse -Force -ErrorAction Stop) {
                if (-not (Test-Path -LiteralPath (Join-Path $folder.FullName 'US') -PathType Container)) { continue }
                $info = Get-FolderInfo -Path $folder.FullName
                $relativePath = $folder.FullName.Substring($sourceRoot.TrimEnd('\').Length).TrimStart('\')
                $inventory += [pscustomobject]@{
                    Name = $relativePath
                    SourcePath = $folder.FullName
                    DestinationPath = Join-Path $destinationRoot $relativePath
                    Bytes = $info.Bytes
                    GB = [math]::Round($info.Bytes / 1GB, 4)
                }
            }
        }
    }
    elseif ($mode -eq 'TopLevelFolders') {
        foreach ($folder in Get-ChildItem -LiteralPath $sourceRoot -Directory -Force | Sort-Object Name) {
            $info = Get-FolderInfo -Path $folder.FullName
            $inventory += [pscustomobject]@{
                Name = $folder.Name
                SourcePath = $folder.FullName
                DestinationPath = Join-Path $destinationRoot $folder.Name
                Bytes = $info.Bytes
                GB = [math]::Round($info.Bytes / 1GB, 4)
            }
        }
    }
    else {
        throw "Unsupported InventoryMode: $mode"
    }
    return @($inventory | Sort-Object Name)
}

function Sync-RootFiles {
    param(
        [Parameter(Mandatory = $true)]$Config,
        [int]$TotalFolders,
        [double]$TotalGB,
        [switch]$WaitForCloud
    )
    if (-not $Config.PSObject.Properties['IncludeRootFiles'] -or -not [bool]$Config.IncludeRootFiles) { return $true }
    $sourceRoot = [string]$Config.SourcePath
    $destinationRoot = [string]$Config.DestinationPath
    $rootFiles = @(Get-ChildItem -LiteralPath $sourceRoot -File -Force -ErrorAction Stop | Sort-Object Name)
    if ($Config.PSObject.Properties['IncludedRootFiles']) {
        $includedRootFiles = @($Config.IncludedRootFiles)
        $rootFiles = @($rootFiles | Where-Object { $_.Name -in $includedRootFiles })
    }
    foreach ($sourceFile in $rootFiles) {
        if (-not (Wait-ForRunPermission -CompletedFolders 0 -TotalFolders $TotalFolders -CompletedGB 0 -TotalGB $TotalGB -CurrentFolders @($sourceFile.Name))) { return $false }
        $destinationFile = Join-Path $destinationRoot $sourceFile.Name
        $needsCopy = -not (Test-Path -LiteralPath $destinationFile -PathType Leaf)
        if (-not $needsCopy) {
            $existing = Get-Item -LiteralPath $destinationFile -Force
            $needsCopy = ($existing.Length -ne $sourceFile.Length) -or ($existing.LastWriteTimeUtc -ne $sourceFile.LastWriteTimeUtc)
        }
        if ($needsCopy) {
            Update-WorkerStatus -Phase 'CopyingRootFiles' -Message ('Copying root file {0}' -f $sourceFile.Name) -CurrentFolders @($sourceFile.Name) -TotalFolders $TotalFolders -TotalGB $TotalGB
            Copy-Item -LiteralPath $sourceFile.FullName -Destination $destinationFile -Force
            Write-Log ('Copied root file {0}' -f $sourceFile.Name)
        }
        & attrib.exe +U -P $destinationFile | Out-Null
        if (-not $WaitForCloud) {
            $destinationInfo = Get-Item -LiteralPath $destinationFile -Force
            if (-not (Test-IsCloudOnlyFile -File $destinationInfo)) {
                Write-Log ('Queued root file for OneDrive upload without blocking patient batches: {0}' -f $sourceFile.Name)
            }
            continue
        }
        while ($true) {
            if (-not (Wait-ForRunPermission -CompletedFolders 0 -TotalFolders $TotalFolders -CompletedGB 0 -TotalGB $TotalGB -CurrentFolders @($sourceFile.Name))) { return $false }
            $destinationInfo = Get-Item -LiteralPath $destinationFile -Force
            if (Test-IsCloudOnlyFile -File $destinationInfo) { break }
            Update-WorkerStatus -Phase 'UploadingRootFiles' -Message ('Waiting for OneDrive to upload root file {0}' -f $sourceFile.Name) -CurrentFolders @($sourceFile.Name) -TotalFolders $TotalFolders -TotalGB $TotalGB
            Start-Sleep -Seconds ([int]$Config.PollSeconds)
            & attrib.exe +U -P $destinationFile | Out-Null
        }
    }
    return $true
}

function Update-WorkerStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string]$Message,
        [string[]]$CurrentFolders = @(),
        [int]$CompletedFolders = 0,
        [int]$TotalFolders = 0,
        [double]$CompletedGB = 0,
        [double]$TotalGB = 0,
        [double]$BatchGB = 0
    )
    $config = Get-Config
    $status = [ordered]@{
        toolVersion = $script:ToolVersion
        workerPid = $PID
        phase = $Phase
        desiredState = Get-ControlState
        message = $Message
        currentFolders = @($CurrentFolders)
        completedFolders = $CompletedFolders
        totalFolders = $TotalFolders
        completedGB = [math]::Round($CompletedGB, 2)
        totalGB = [math]::Round($TotalGB, 2)
        batchGB = [math]::Round($BatchGB, 2)
        freeSpaceGB = Get-FreeSpaceGB -Path ([string]$config.DestinationPath)
        reserveFreeGB = [double]$config.ReserveFreeGB
        updatedAt = (Get-Date).ToString('o')
        logPath = $script:LogPath
    }
    Write-JsonFile -Path $script:StatusPath -Value $status
}

function Wait-ForRunPermission {
    param(
        [int]$CompletedFolders,
        [int]$TotalFolders,
        [double]$CompletedGB,
        [double]$TotalGB,
        [string[]]$CurrentFolders = @()
    )
    while ($true) {
        $desired = Get-ControlState
        if ($desired -eq 'stopped') { return $false }
        if ($desired -eq 'running') { return $true }
        Update-WorkerStatus -Phase 'Paused' -Message 'Paused. OneDrive may continue uploading files already staged.' -CurrentFolders $CurrentFolders -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
        Start-Sleep -Seconds 2
    }
}

function Test-OneDriveRunning {
    return ($null -ne (Get-Process -Name 'OneDrive' -ErrorAction SilentlyContinue | Select-Object -First 1))
}

function Invoke-RobocopyFolder {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)]$Config,
        [int]$CompletedFolders,
        [int]$TotalFolders,
        [double]$CompletedGB,
        [double]$TotalGB
    )
    $robocopyLog = Join-Path $script:StateDirectory 'robocopy.log'
    $arguments = @(
        ('"{0}"' -f $SourcePath),
        ('"{0}"' -f $DestinationPath),
        '/E', '/Z', '/J', ('/MT:{0}' -f [int]$Config.RobocopyThreads),
        '/COPY:DAT', '/DCOPY:DAT', '/R:3', '/W:5', '/XJ',
        '/NP', '/NFL', '/NDL', ('/LOG+:{0}' -f $robocopyLog)
    )

    while ($true) {
        if (-not (Wait-ForRunPermission -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB -CurrentFolders @((Split-Path -Leaf $SourcePath)))) {
            return 'stopped'
        }
        Update-WorkerStatus -Phase 'Copying' -Message ('Copying/resuming {0}' -f (Split-Path -Leaf $SourcePath)) -CurrentFolders @((Split-Path -Leaf $SourcePath)) -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
        Write-Log ('Starting robocopy for {0}' -f (Split-Path -Leaf $SourcePath))
        $process = Start-Process -FilePath 'robocopy.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru

        while (-not $process.HasExited) {
            Start-Sleep -Seconds 2
            $process.Refresh()
            $desired = Get-ControlState
            if ($desired -eq 'paused' -or $desired -eq 'stopped') {
                Write-Log ('Interrupting robocopy for {0}: requested {1}' -f (Split-Path -Leaf $SourcePath), $desired)
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                $process.WaitForExit()
                if ($desired -eq 'stopped') { return 'stopped' }
                break
            }
        }

        if (-not $process.HasExited) { continue }
        $exitCode = $process.ExitCode
        if ($exitCode -le 7) {
            Write-Log ('Robocopy completed for {0} with code {1}' -f (Split-Path -Leaf $SourcePath), $exitCode)
            return 'copied'
        }
        Write-Log ('Robocopy failed for {0} with code {1}; retrying' -f (Split-Path -Leaf $SourcePath), $exitCode)
        Update-WorkerStatus -Phase 'Retrying' -Message ('Copy error for {0} (robocopy code {1}). Retrying.' -f (Split-Path -Leaf $SourcePath), $exitCode) -CurrentFolders @((Split-Path -Leaf $SourcePath)) -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
        Start-Sleep -Seconds 15
    }
}

function Request-FreeUpSpace {
    param([Parameter(Mandatory = $true)][string]$FolderPath)
    Write-Log ('Requesting OneDrive Free up space for {0}' -f (Split-Path -Leaf $FolderPath))
    $wildcardPath = Join-Path $FolderPath '*'
    & attrib.exe +U -P $wildcardPath /S /D | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Log ('attrib returned code {0} for {1}' -f $LASTEXITCODE, (Split-Path -Leaf $FolderPath))
    }
}

function Wait-ForCloudFolder {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)]$Config,
        [int]$CompletedFolders,
        [int]$TotalFolders,
        [double]$CompletedGB,
        [double]$TotalGB
    )
    $folderName = Split-Path -Leaf $SourcePath
    $lastRequest = [datetime]::MinValue
    while ($true) {
        if (-not (Wait-ForRunPermission -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB -CurrentFolders @($folderName))) {
            return $false
        }

        if (-not (Test-OneDriveRunning)) {
            Update-WorkerStatus -Phase 'WaitingForOneDrive' -Message 'OneDrive is not running. Start OneDrive; this tool will keep waiting.' -CurrentFolders @($folderName) -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
            Start-Sleep -Seconds ([int]$Config.PollSeconds)
            continue
        }

        if (((Get-Date) - $lastRequest).TotalSeconds -ge 60) {
            Request-FreeUpSpace -FolderPath $DestinationPath
            $lastRequest = Get-Date
        }

        $cloudState = Get-CloudFolderState -SourcePath $SourcePath -DestinationPath $DestinationPath
        if ($cloudState.Ready) {
            Write-Log ('Cloud-only confirmation complete for {0}' -f $folderName)
            return $true
        }

        $message = 'Waiting for OneDrive: {0} local/pending files remain in {1}.' -f $cloudState.LocalFiles, $folderName
        if ($cloudState.SourceFiles -ne $cloudState.DestinationFiles) {
            $message = 'Waiting/validating {0}: source files {1}, destination files {2}.' -f $folderName, $cloudState.SourceFiles, $cloudState.DestinationFiles
        }
        Update-WorkerStatus -Phase 'Uploading' -Message $message -CurrentFolders @($folderName) -CompletedFolders $CompletedFolders -TotalFolders $TotalFolders -CompletedGB $CompletedGB -TotalGB $TotalGB
        Start-Sleep -Seconds ([int]$Config.PollSeconds)
    }
}

function Select-NextBatch {
    param(
        [Parameter(Mandatory = $true)][object[]]$Pending,
        [Parameter(Mandatory = $true)]$Config
    )
    $freeGB = Get-FreeSpaceGB -Path ([string]$Config.DestinationPath)
    $availableGB = $freeGB - [double]$Config.ReserveFreeGB
    if ($availableGB -le 0) { return @() }

    $batch = @()
    $batchGB = 0.0
    foreach ($item in $Pending) {
        $itemGB = [double]$item.GB
        $fitsDisk = (($batchGB + $itemGB) -le $availableGB)
        $fitsBatch = (($batchGB + $itemGB) -le [double]$Config.MaxBatchGB)
        if ($batch.Count -eq 0 -and $fitsDisk) {
            $batch += $item
            $batchGB += $itemGB
            if ($itemGB -ge [double]$Config.MaxBatchGB) { break }
        }
        elseif ($batch.Count -lt [int]$Config.MaxFoldersPerBatch -and $fitsDisk -and $fitsBatch) {
            $batch += $item
            $batchGB += $itemGB
        }
        if ($batch.Count -ge [int]$Config.MaxFoldersPerBatch) { break }
    }
    return @($batch)
}

function Invoke-Worker {
    Initialize-StateDirectory
    # All configured datasets share this mutex so two workers cannot independently
    # consume the same C: staging reserve at the same time.
    $mutex = New-Object System.Threading.Mutex($false, 'Local\OneDriveDatasetBatchSyncGlobalWorker')
    $hasMutex = $false
    try {
        Set-Content -LiteralPath $script:WorkerPidPath -Value $PID -Encoding ASCII
        $config = Get-Config
        $sourceRoot = [string]$config.SourcePath
        $destinationRoot = [string]$config.DestinationPath
        while (-not $hasMutex) {
            $hasMutex = $mutex.WaitOne(0, $false)
            if ($hasMutex) { break }
            if ((Get-ControlState) -eq 'stopped') {
                Update-WorkerStatus -Phase 'Stopped' -Message 'Stopped while waiting for the other dataset sync to finish or pause.'
                return
            }
            Update-WorkerStatus -Phase 'WaitingForOtherDataset' -Message 'Waiting: another dataset sync currently owns the shared C: staging space. Stop that worker to switch datasets.'
            Start-Sleep -Seconds 5
        }
        if (-not (Test-Path -LiteralPath $sourceRoot)) { throw "Source is unavailable: $sourceRoot" }
        if (-not (Test-Path -LiteralPath $destinationRoot)) { New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null }

        Write-Log ('Worker started (PID {0})' -f $PID)
        Update-WorkerStatus -Phase 'Scanning' -Message 'Scanning source folders and existing OneDrive placeholders.'
        $inventory = @(Get-WorkInventory -Config $config)
        $totalBytes = [long](($inventory | Measure-Object -Property Bytes -Sum).Sum)
        $totalFolders = $inventory.Count
        $totalGB = $totalBytes / 1GB
        if (-not (Sync-RootFiles -Config $config -TotalFolders $totalFolders -TotalGB $totalGB)) {
            Update-WorkerStatus -Phase 'Stopped' -Message 'Stopped safely while processing root files.' -TotalFolders $totalFolders -TotalGB $totalGB
            return
        }

        while ($true) {
            if (-not (Wait-ForRunPermission -CompletedFolders 0 -TotalFolders $totalFolders -CompletedGB 0 -TotalGB $totalGB)) {
                Update-WorkerStatus -Phase 'Stopped' -Message 'Stopped safely. Run Start/Resume later to continue.' -TotalFolders $totalFolders -TotalGB $totalGB
                Write-Log 'Worker stopped by user'
                return
            }

            $completed = @()
            $pendingExisting = @()
            $pendingNew = @()
            $completedBytes = 0L
            Update-WorkerStatus -Phase 'Scanning' -Message 'Checking which folders are already uploaded and space-free.' -TotalFolders $totalFolders -TotalGB $totalGB
            foreach ($item in $inventory) {
                if (Test-Path -LiteralPath $item.DestinationPath) {
                    $cloudState = Get-CloudFolderState -SourcePath $item.SourcePath -DestinationPath $item.DestinationPath
                    if ($cloudState.Ready) {
                        $completed += $item
                        $completedBytes += $item.Bytes
                    }
                    else {
                        $pendingExisting += $item
                    }
                }
                else {
                    $pendingNew += $item
                }
            }

            $completedFolders = $completed.Count
            $completedGB = $completedBytes / 1GB
            if ($completedFolders -eq $totalFolders) {
                if (-not (Sync-RootFiles -Config $config -TotalFolders $totalFolders -TotalGB $totalGB -WaitForCloud)) {
                    Update-WorkerStatus -Phase 'Stopped' -Message 'Stopped safely while verifying root files.' -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                    return
                }
                Set-ControlState -DesiredState 'stopped'
                Update-WorkerStatus -Phase 'Completed' -Message ('All {0} folders are uploaded and local space has been freed.' -f $totalFolders) -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                Write-Log ('COMPLETED: all {0} folders are cloud-only' -f $totalFolders)
                return
            }

            # Existing local/pending folders always go first. This adopts manual copies safely.
            if ($pendingExisting.Count -gt 0) {
                $batch = @($pendingExisting | Select-Object -First 1)
            }
            else {
                $batch = @(Select-NextBatch -Pending $pendingNew -Config $config)
            }

            if ($batch.Count -eq 0) {
                $message = 'Not enough free C: space for the next folder while preserving the {0} GB reserve.' -f $config.ReserveFreeGB
                Update-WorkerStatus -Phase 'WaitingForSpace' -Message $message -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                Write-Log $message
                Start-Sleep -Seconds 30
                continue
            }

            $batchGB = (($batch | Measure-Object -Property GB -Sum).Sum)
            $batchNames = @($batch | ForEach-Object { $_.Name })
            Update-WorkerStatus -Phase 'PreparingBatch' -Message ('Preparing batch: {0}' -f ($batchNames -join ', ')) -CurrentFolders $batchNames -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB -BatchGB $batchGB

            foreach ($item in $batch) {
                $result = Invoke-RobocopyFolder -SourcePath $item.SourcePath -DestinationPath $item.DestinationPath -Config $config -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                if ($result -eq 'stopped') {
                    Update-WorkerStatus -Phase 'Stopped' -Message 'Stopped safely during copy. Robocopy will resume the partial file next time.' -CurrentFolders @($item.Name) -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                    Write-Log 'Worker stopped during copy'
                    return
                }
            }

            foreach ($item in $batch) {
                $ready = Wait-ForCloudFolder -SourcePath $item.SourcePath -DestinationPath $item.DestinationPath -Config $config -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                if (-not $ready) {
                    Update-WorkerStatus -Phase 'Stopped' -Message 'Stopped safely. OneDrive may continue the current upload.' -CurrentFolders @($item.Name) -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
                    Write-Log 'Worker stopped while waiting for OneDrive'
                    return
                }
                $completedFolders++
                $completedGB += [double]$item.GB
                Update-WorkerStatus -Phase 'BatchComplete' -Message ('Uploaded and freed local space for {0}' -f $item.Name) -CurrentFolders @($item.Name) -CompletedFolders $completedFolders -TotalFolders $totalFolders -CompletedGB $completedGB -TotalGB $totalGB
            }
        }
    }
    catch {
        try {
            Write-Log ('FATAL [{0}]: {1}' -f $_.Exception.GetType().FullName, $_.Exception.Message)
            if ($_.ScriptStackTrace) { Write-Log ('STACK: {0}' -f ($_.ScriptStackTrace -replace "`r?`n", ' | ')) }
            $config = Get-Config
            $priorStatus = Read-JsonFile -Path $script:StatusPath
            $priorCurrentFolders = @()
            $priorCompletedFolders = 0
            $priorTotalFolders = 0
            $priorCompletedGB = 0
            $priorTotalGB = 0
            $priorBatchGB = 0
            if ($null -ne $priorStatus) {
                $priorCurrentFolders = @($priorStatus.currentFolders)
                $priorCompletedFolders = [int]$priorStatus.completedFolders
                $priorTotalFolders = [int]$priorStatus.totalFolders
                $priorCompletedGB = [double]$priorStatus.completedGB
                $priorTotalGB = [double]$priorStatus.totalGB
                $priorBatchGB = [double]$priorStatus.batchGB
            }
            $status = [ordered]@{
                toolVersion = $script:ToolVersion
                workerPid = $PID
                phase = 'Error'
                desiredState = Get-ControlState
                message = $_.Exception.Message
                currentFolders = $priorCurrentFolders
                completedFolders = $priorCompletedFolders
                totalFolders = $priorTotalFolders
                completedGB = $priorCompletedGB
                totalGB = $priorTotalGB
                batchGB = $priorBatchGB
                freeSpaceGB = Get-FreeSpaceGB -Path ([string]$config.DestinationPath)
                reserveFreeGB = [double]$config.ReserveFreeGB
                updatedAt = (Get-Date).ToString('o')
                logPath = $script:LogPath
            }
            Write-JsonFile -Path $script:StatusPath -Value $status
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
    $form.Size = New-Object System.Drawing.Size(760, 535)
    $form.MinimumSize = New-Object System.Drawing.Size(700, 500)
    $form.StartPosition = 'CenterScreen'
    $form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

    $title = New-Object System.Windows.Forms.Label
    $title.Text = ('{0} -> Technion OneDrive' -f $script:DatasetLabel)
    $title.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 16)
    $title.AutoSize = $true
    $title.Location = New-Object System.Drawing.Point(20, 18)
    $form.Controls.Add($title)

    $phaseLabel = New-Object System.Windows.Forms.Label
    $phaseLabel.Text = 'State: Not started'
    $phaseLabel.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 11)
    $phaseLabel.AutoSize = $true
    $phaseLabel.Location = New-Object System.Drawing.Point(22, 62)
    $form.Controls.Add($phaseLabel)

    $messageLabel = New-Object System.Windows.Forms.Label
    $messageLabel.Text = 'Use Start / Resume to begin.'
    $messageLabel.Location = New-Object System.Drawing.Point(22, 92)
    $messageLabel.Size = New-Object System.Drawing.Size(690, 45)
    $form.Controls.Add($messageLabel)

    $progress = New-Object System.Windows.Forms.ProgressBar
    $progress.Location = New-Object System.Drawing.Point(25, 143)
    $progress.Size = New-Object System.Drawing.Size(690, 25)
    $progress.Minimum = 0
    $progress.Maximum = 1
    $form.Controls.Add($progress)

    $detailsLabel = New-Object System.Windows.Forms.Label
    $detailsLabel.Text = 'Progress information will appear here.'
    $detailsLabel.Location = New-Object System.Drawing.Point(22, 178)
    $detailsLabel.Size = New-Object System.Drawing.Size(690, 48)
    $form.Controls.Add($detailsLabel)

    $startButton = New-Object System.Windows.Forms.Button
    $startButton.Text = 'Start / Resume'
    $startButton.Location = New-Object System.Drawing.Point(25, 235)
    $startButton.Size = New-Object System.Drawing.Size(150, 40)
    $form.Controls.Add($startButton)

    $pauseButton = New-Object System.Windows.Forms.Button
    $pauseButton.Text = 'Pause'
    $pauseButton.Location = New-Object System.Drawing.Point(190, 235)
    $pauseButton.Size = New-Object System.Drawing.Size(120, 40)
    $form.Controls.Add($pauseButton)

    $stopButton = New-Object System.Windows.Forms.Button
    $stopButton.Text = 'Stop'
    $stopButton.Location = New-Object System.Drawing.Point(325, 235)
    $stopButton.Size = New-Object System.Drawing.Size(120, 40)
    $form.Controls.Add($stopButton)

    $logButton = New-Object System.Windows.Forms.Button
    $logButton.Text = 'Open log folder'
    $logButton.Location = New-Object System.Drawing.Point(460, 235)
    $logButton.Size = New-Object System.Drawing.Size(130, 40)
    $form.Controls.Add($logButton)

    $destinationButton = New-Object System.Windows.Forms.Button
    $destinationButton.Text = 'Open destination'
    $destinationButton.Location = New-Object System.Drawing.Point(600, 235)
    $destinationButton.Size = New-Object System.Drawing.Size(120, 40)
    $form.Controls.Add($destinationButton)

    $logBox = New-Object System.Windows.Forms.TextBox
    $logBox.Location = New-Object System.Drawing.Point(25, 295)
    $logBox.Size = New-Object System.Drawing.Size(690, 155)
    $logBox.Multiline = $true
    $logBox.ReadOnly = $true
    $logBox.ScrollBars = 'Vertical'
    $logBox.Font = New-Object System.Drawing.Font('Consolas', 8.5)
    $form.Controls.Add($logBox)

    $note = New-Object System.Windows.Forms.Label
    $note.Text = 'Closing this window does not stop the worker. Pause or Stop first if desired.'
    $note.Location = New-Object System.Drawing.Point(22, 462)
    $note.AutoSize = $true
    $form.Controls.Add($note)

    $refreshAction = {
        try {
            $status = Read-JsonFile -Path $script:StatusPath
            if ($null -ne $status) {
                $phaseLabel.Text = 'State: {0}' -f $status.phase
                $messageLabel.Text = [string]$status.message
                $maximum = [math]::Max(1, [int]$status.totalFolders)
                $progress.Maximum = $maximum
                $progress.Value = [math]::Min($maximum, [int]$status.completedFolders)
                $folderText = if ($status.currentFolders.Count -gt 0) { $status.currentFolders -join ', ' } else { '-' }
                $detailsLabel.Text = ('Folders: {0}/{1}    Data: {2:N1}/{3:N1} GB    Free C:: {4:N1} GB (reserve {5:N0})`r`nCurrent: {6}' -f [int]$status.completedFolders, [int]$status.totalFolders, [double]$status.completedGB, [double]$status.totalGB, [double]$status.freeSpaceGB, [double]$status.reserveFreeGB, $folderText)
            }
            if (Test-Path -LiteralPath $script:LogPath) {
                $logBox.Lines = @(Get-Content -LiteralPath $script:LogPath -Tail 12 -ErrorAction SilentlyContinue)
                $logBox.SelectionStart = $logBox.TextLength
                $logBox.ScrollToCaret()
            }
        }
        catch { }
    }

    $startButton.Add_Click({
        Set-ControlState -DesiredState 'running'
        Start-WorkerProcess | Out-Null
        Write-Log 'Start/Resume requested from dashboard'
        & $refreshAction
    })
    $pauseButton.Add_Click({
        Set-ControlState -DesiredState 'paused'
        Write-Log 'Pause requested from dashboard'
        & $refreshAction
    })
    $stopButton.Add_Click({
        Set-ControlState -DesiredState 'stopped'
        Write-Log 'Stop requested from dashboard'
        & $refreshAction
    })
    $logButton.Add_Click({
        Initialize-StateDirectory
        Start-Process explorer.exe -ArgumentList ('"{0}"' -f $script:StateDirectory)
    })
    $destinationButton.Add_Click({
        $config = Get-Config
        Start-Process explorer.exe -ArgumentList ('"{0}"' -f [string]$config.DestinationPath)
    })

    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 2000
    $timer.Add_Tick($refreshAction)
    $timer.Start()
    & $refreshAction
    [void]$form.ShowDialog()
    $timer.Stop()
}

Initialize-StateDirectory
switch ($Mode) {
    'Dashboard' { Show-Dashboard }
    'Start' {
        Set-ControlState -DesiredState 'running'
        $workerPid = Start-WorkerProcess
        Write-Output "Started/resumed worker PID $workerPid"
    }
    'Resume' {
        Set-ControlState -DesiredState 'running'
        $workerPid = Start-WorkerProcess
        Write-Output "Started/resumed worker PID $workerPid"
    }
    'Pause' {
        Set-ControlState -DesiredState 'paused'
        Write-Output 'Pause requested'
    }
    'Stop' {
        Set-ControlState -DesiredState 'stopped'
        Write-Output 'Stop requested'
    }
    'Status' {
        $status = Read-JsonFile -Path $script:StatusPath
        if ($null -eq $status) { Write-Output 'Not started' } else { $status | ConvertTo-Json -Depth 8 }
    }
    'Worker' { Invoke-Worker }
}
