"""GE-only held-out evaluation for corrected first-R-to-second-R PanEcho."""

from pathlib import Path

import ichilov3_rwave_panecho_ge_eval as ge


ge.ROOT = Path(r"D:\DS\ichilov3_rwave_first_cycle_panecho_20261005")
ge.NEW = ge.ROOT / "panecho_visits.npz"

if __name__ == "__main__":
    ge.main()
    report = ge.ROOT / "ge_only" / "results.md"
    report.write_text(report.read_text(encoding="utf-8").replace(
        "Fourteen source cines span multiple cycles between first and last R marker.",
        "All R-wave cines use only the cycle between the first and second R markers; 14 cines were corrected from the earlier first-to-last trial."
    ), encoding="utf-8")
