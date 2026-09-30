# Ichilov apical-view selector

Local browser app for reviewing source echo cine loops and assigning A2C, A3C, and A4C to visits whose three clips were not resolved by the DICOM audit. It reads the existing `ichilov3_20260923` audit and opens source DICOMs from `E:\ichilov3` **read-only**. No DICOM files are copied, moved, or modified.

## Start

In PowerShell:

```powershell
& 'D:\us\.venv\Scripts\python.exe' 'D:\us\ichilov3_view_selector\server.py'
```

Open <http://127.0.0.1:8765/> on the same computer. Leave the PowerShell window open while reviewing. Use `Ctrl+C` to stop the server. The app binds to `127.0.0.1` only. Source DICOM frames may contain burned-in patient identifiers, so do not expose the app on a network or use screenshots as anonymized data.

The default queue contains the 41 visits without a TOMTEC bookmark. **All unresolved** also includes visits with incomplete or conflicting bookmark matches. **All visits** is available for comparison; fully resolved TOMTEC visits cannot be saved over in this app. Bookmark visits have a separate **Original analysis clips** section showing the recovered or unresolved status of each view. It is separate from model suggestions and your choices. **Use linked clip** explicitly adds a recovered clip to your choices and records `tomtec` provenance; it does not preselect anything.

Click **PLAY CINE** on a candidate to watch the full DICOM loop. The player loops, has a scrub bar, and offers 0.25×, 0.5×, and 1× playback speed. Use **Frame by frame** for still inspection if needed. Assign a different source file to each of A2C, A3C, and A4C. Once all three are marked, the app saves the review automatically; later replacements are saved too. Enter a reviewer name once so saved choices are attributable. Partial decisions and note edits can still be saved with **Save review**. If no appropriate three-view combination exists, click **Mark whole visit invalid**; this clears the selections and immediately saves an `invalid` status. A missing note receives a standard reason, which you can edit later. Recovered bookmark clips appear in their own section; the selection slots contain only your explicit choices.

Click **Analyze this visit** to generate model suggestions. The local EchoPrime view classifier scores five DICOM frames per cine after cropping to the declared 2D ultrasound tissue region. A separate technical screen favors B-mode loops with adequate visible tissue, motion, duration, and image detail. The app suggests at most one distinct clip per view and abstains when view probability or frame agreement is too low. It shows alternative candidates for review. Click **Use this** or **Use available suggestions in empty slots** to accept candidates provisionally; you can replace any choice from the cine library. Accepting all three triggers the same automatic save. The export records whether each saved choice came from a model suggestion or a manual override. Generating suggestions alone never changes a saved decision.

The technical screen does **not** establish that the true LV apex is visible, that a view is free of foreshortening, or that every endocardial border can be tracked. Those require your inspection of the full loop. A spot check against 24 TOMTEC-linked source clips found 23 correct top-1 view labels; this is a small, non-independent check of view labels, not clinical validation of the selected clip or strain quality. Predictions and suggestions are stored separately under `D:\us\output\dicom_prediction\ichilov3_manual_selection` and source DICOMs remain untouched.

The app groups clips by the DICOM `StudyDate`, not by the folder's date label. It flags clips whose folder date differs from `StudyDate`. Two visits in this audit have date-mismatched folders and therefore no matching cine loops under their report dates; they remain in the queue for triage.

## Results

Manual decisions are stored at `D:\us\output\dicom_prediction\ichilov3_manual_selection\decisions.json`. Every subsequent save first creates a timestamped `decisions.backup.*.json` in the same directory, then atomically replaces the current snapshot. **Export decisions** downloads a CSV with the selected source paths, SOP Instance UIDs, Study Instance UIDs, reviewer, status, and notes. This file is separate from the audit and source DICOMs; the app does not assert that a manual selection matches the original strain analysis.

The app creates an H.264 preview on demand in `D:\us\output\dicom_prediction\ichilov3_manual_selection\cine_cache`. It uses the DICOM frame timing when present and never writes to `E:\ichilov3`. A first play can take several seconds while the clip is prepared; repeat plays use the cached preview. Some unusual transfer syntaxes may fail to preview; use **Frame by frame**, record the problem in a note, and retain the original source files for later decoding work.

Clicking outside the cine popup closes it. The × button and Escape also work.
