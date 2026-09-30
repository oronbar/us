# Ichilov crop review

Mobile app for the 303 clips flagged during spatial processing. Swipe right for
good, left for bad. Explicit buttons provide the same actions. Compare the
current full cine to original pixels inside the DICOM tissue region.

Each flagged cine has a stable review index from 1 to 303. Go to # switches to
the full review queue so already-reviewed entries can be reopened by index.
Find another cine from this view ranks up to five B-mode alternatives within
the same visit folder and exact DICOM Study UID, excluding already-selected
SOPs. Model view probabilities and technical proxies are screening aids, not
validated strain-quality scores. Short-loop warnings remain visible.

After previewing a suggestion, Use this cine & re-encode creates a new crop and
both embeddings in a revision. The effective manifest records the replacement
source SOP/path and keeps original source/selection provenance separately; it
does not claim the replacement was analyzed in TOMTEC. Base selections remain
unchanged. Pending or bad crop decisions remain excluded.

Run from D:\us:

```powershell
.venv\Scripts\python.exe ichilov3_crop_reviewer\server.py
```

Local URL: http://127.0.0.1:8770. Tailscale Serve can proxy this loopback server
privately to the tailnet; no Funnel or public hosting is used.

## Storage and repair

- Decisions: D:\DS\ichilov3_crop_review\reviews.sqlite3, with an append-only
  history table. Decisions survive closing the browser and restarting the app.
- Active manifest: D:\DS\ichilov3_crop_review\reviewed_crop_manifest.json.
  Every decision updates it. Pending flagged and rejected clips are marked
  ineligible; unflagged clips and good/repaired decisions are eligible for crop
  quality only, not necessarily for outcome modeling.
- Repaired crops: revisions\<file_id>\<revision>\crop.npz with all source frames
  and original frame indices. Each revision has its own metadata and new
  EchoPrime and PanEcho embeddings. Base crops and base embedding arrays remain
  unchanged. Future training must use the active paths in the reviewed manifest.
- Browser videos: disposable H.264 previews in media. Display timing uses
  FrameTime/CineRate where available; these are previews, not new source data.

Restore uses the declared DICOM tissue rectangle without per-frame connected
component suppression. Alternatively draw a rectangle within that tissue region
and check it across frames. Crops are square-padded and resized to 518 pixels.
Save fix & re-encode creates a version, processes both frozen encoders, and marks
the clip repaired only after both succeed. A failed repair preserves earlier data.

For intrinsically poor acquisitions, changing crop geometry cannot repair the
video. Mark it bad and keep it excluded, or select a different source cine in the
view-selection workflow and rerun processing for that selection.

Tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s ichilov3_crop_reviewer -p test_server.py
```

## Smooth mobile review

The cine follows the pointer without easing during a horizontal drag. Large GOOD/BAD stamps and a colored outline show direction. Release beyond the threshold commits the decision and flies the cine off-screen; short or vertical gestures snap back. A failed save restores the current card. Reduced-motion settings shorten the exit animation.

The next three current-crop previews are downloaded into a bounded browser buffer, using two background workers after the visible cine is ready. Moving or jumping releases obsolete clips; crop revision IDs prevent stale repaired previews. At most five small preview blobs are retained (16 MiB per clip); larger cines stream through the browser normally. The buffering counter indicates readiness. A decision is persisted before advancing; the confirmation response updates counts without fetching the entire state again.

Buffer tests: `node --test ichilov3_crop_reviewer/test_cine_buffer.cjs`.
