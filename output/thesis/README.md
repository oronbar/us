# English thesis draft

The readable manuscript is `../pdf/thesis_draft.pdf`. The editable resolved version is `thesis_draft.md`; references, evidence notes, the source manifest and the aggregate figure accompany it.

## Rebuilding

The authoring scripts currently use the original Windows workspace at `D:/us`, the two supplied proposal PDF paths and Times New Roman fonts under `C:/Windows/Fonts`. Update these paths when moving to another machine.

From the repository root, run:

```powershell
python tmp/thesis_harvest.py
python tmp/thesis_review/audit.py
python tmp/thesis_review/build_thesis.py
```

Python dependencies include pypdf, pandas, pyarrow, scikit-learn, pymupdf, Pillow, matplotlib and reportlab. The audit script checks saved data and predictions; it does not retrain models. The builder reads the manuscript template in `tmp/thesis_review/manuscript.md`, inserts tables and references, and regenerates the PDF and resolved Markdown. Edit that template for changes intended to survive a rebuild.

The original proposal PDFs and authorized research datasets must be available locally to reproduce the full harvest and audit. Generated extraction caches and page previews are excluded from Git. Existing numerical research outputs are not modified by these scripts.

After rebuilding, render the PDF with Poppler and inspect its pages before distribution. The committed draft was visually checked at 28 pages. Consult `evidence_audit.md` for the review scope, methodological qualifications and work remaining before thesis submission.
