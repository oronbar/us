# SZMC2 OneDrive Batch Sync

This Windows tool stages the 282 folders from `F:\DS\SZMC2` into the Technion
OneDrive folder without requiring enough C: space for the complete 1.71 TB data
set.

## Use

1. Keep the external F: drive connected and make sure the OneDrive desktop app
   is signed in and running.
2. Double-click **Start SZMC2 Sync Tool.cmd**.
3. Click **Start / Resume**.

The window can be closed while the worker continues. Open the same CMD file
again to see status or to pause/stop it.

## What the controls mean

- **Start / Resume** starts a new worker or resumes the paused worker.
- **Pause** interrupts an active Robocopy operation in restartable mode. OneDrive
  can continue uploading files already copied. Resume continues safely.
- **Stop** exits the worker safely. A partially copied file is resumed on the
  next start. It does not stop the OneDrive application.

## Safety behavior

- The source drive is read-only to this tool: source folders are never moved,
  edited, or deleted.
- Existing destination folders (including the current manual copy) are adopted
  and completed with Robocopy rather than deleted or blindly recopied.
- No destination item is deleted. After a folder is fully uploaded, the tool
  requests OneDrive **Free up space** (`attrib +U -P`) and waits until every file
  is a cloud-only placeholder before proceeding.
- The tool keeps at least 30 GB free on C: before selecting a new adaptive batch.
- Completion means all source folder/file counts and byte totals match and every
  destination file has OneDrive's cloud recall + unpinned attributes.

## Command-line controls (optional)

From PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\SZMC2Sync.ps1 -Mode Start
powershell -NoProfile -ExecutionPolicy Bypass -File .\SZMC2Sync.ps1 -Mode Pause
powershell -NoProfile -ExecutionPolicy Bypass -File .\SZMC2Sync.ps1 -Mode Stop
powershell -NoProfile -ExecutionPolicy Bypass -File .\SZMC2Sync.ps1 -Mode Status
```

Runtime status and logs are stored in:

`%LOCALAPPDATA%\SZMC2OneDriveBatchSync`

## Ichilov2 version

Double-click **Start Ichilov2 Sync Tool.cmd** to sync the dataset on `E:\` to
`C:\Users\Oron\OneDrive - Technion\DS\Ichilov2`.

The Ichilov2 configuration preserves the five `echo_*` trees and processes 400
patient folders as disk-safe work units. It also copies
`anonymization_mapping_hashed.xlsx`. Windows system folders, the Recycle Bin,
and the external-drive manuals are intentionally excluded.

SZMC2, Ichilov2, and Ichilov3 have separate dashboards, controls, state, and
logs. They share one staging lock so only one can consume C: space at a time.
To switch datasets, **Stop** the active worker, then start the other one. A
paused worker keeps ownership of the staging lock so it can resume in place.

## Ichilov3 version

Double-click **Start Ichilov3 Sync Tool.cmd** to sync the 103 patient folders
under `E:\Ichilov3` to
`C:\Users\Oron\OneDrive - Technion\DS\Ichilov3`. The root Excel mapping file
is included. Ichilov3 has its own dashboard, controls, status, and log directory
at `%LOCALAPPDATA%\Ichilov3OneDriveBatchSync`, while sharing the same staging
lock and C: safety reserve with the other dataset tools.

## Important

Do not choose **Always keep on this device** for the destination tree. Files On-
Demand must be enabled for the tool to reclaim C: space. Pause or stop the tool
before disconnecting the F: drive. If the drive is disconnected unexpectedly,
the worker reports an error; reconnect it and use Start / Resume.
