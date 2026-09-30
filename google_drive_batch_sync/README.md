# SZMC2 to Google Drive batch sync

This tool copies `F:\DS\SZMC2` to
`G:\האחסון שלי\DS\SZMC2` through Google Drive for desktop's streamed G: mount.

## Start

1. Confirm Google Drive for desktop is running and **My Drive** is set to
   **Stream files**, not Mirror files.
2. Stop the SZMC2/Ichilov2 OneDrive batch worker if one is active.
3. Double-click **Start Google Drive SZMC2 Sync Tool.cmd**.
4. Click **Start / Resume**.

The worker continues if the dashboard is closed. Reopen the CMD file to view
status or use Pause/Stop.

## Safety and resume behavior

- The F: source is never modified or deleted.
- Robocopy uses restartable mode (`/Z`) and verifies each destination folder by
  file count and total logical bytes.
- Copying is deliberately throttled with a 9 ms inter-packet gap, approximately
  7-8 MB/s for this large-file dataset. This remains below a 100 Mbit/s uplink
  and limits growth of Google Drive's local write cache.
- The worker keeps at least 30 GB free on C:. It interrupts a copy and waits if
  the reserve is reached; Google Drive can continue draining its cache.
- Existing matching folders are skipped on restart. No destination files are
  deleted.
- All dataset tools share one staging lock, so the Google and OneDrive workers
  cannot consume C: cache space simultaneously. A paused worker retains the
  lock; Stop it before switching tools.

## Completion semantics

`DriveMountComplete` means every source folder matches the Google Drive G:
mount. Google Drive for desktop may still have a small final background upload
queue. Keep it running until its tray menu says **Up to date** before treating
the remote cloud upload as final.

Runtime status and logs are stored under:

`%LOCALAPPDATA%\SZMC2GoogleDriveBatchSync`

Before starting, verify that the Google account has enough cloud quota for the
approximately 1.71 TiB dataset.
