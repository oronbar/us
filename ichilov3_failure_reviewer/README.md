# Ichilov 3 GLS cine failure review

Open `http://oron-desktop.tailbe45ac.ts.net:8771/` from a device connected to the same Tailscale network. The server is currently bound to the desktop's Tailscale IPv4 address on port 8771.

The queue defaults to Philips visits sorted by absolute held-out Mid-GLS error. Switch to Endo-GLS or another vendor in the controls. Each visit has three complete cropped cines; `Original frame` decodes the original DICOM cine for comparison. Mobile shows one view at a time, desktop shows three.

Save `No visible issue`, `Suspected issue`, or `Uncertain`, with optional reason tags and notes. Decisions are stored in `D:\DS\ichilov3_temporal_trial_20260927\vendor_failure_analysis\cine_reviews.sqlite3` and exported through the page as CSV. This review does not alter DICOMs, crops, embeddings, predictions, or training eligibility. Generated MP4 files are derived previews under `review_media`.

When you select `Suspected issue`, choose A2C/A3C/A4C and press `Find ranked alternatives`. The app ranks unselected cines from the same DICOM study using the existing EchoPrime view classifier and quality proxy. It shows the highest-ranked alternative first and up to four lower-ranked options. Preview the full cine, then press `Use previewed cine` to save a replacement mapping. The suspected-issue review is saved at the same time. `Restore original selection` removes the mapping; all changes are recorded in history tables. The export includes replacement paths and SOP identifiers for each view. Predictions on the page still belong to the original three-cine set until the approved replacements are cropped, encoded and re-evaluated.

To restart the app if needed:

```powershell
& 'D:\us\.venv\Scripts\python.exe' 'D:\us\ichilov3_failure_reviewer\server.py' --host 100.73.61.63 --port 8771
```

The address may change if Tailscale assigns a different IPv4 address. The DNS hostname normally remains stable. Re-run `tailscale ip -4` if the server cannot bind.
