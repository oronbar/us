"""Local EchoPrime view suggestions with conservative strain-suitability screening.

Predictions are suggestions for human review, not validated strain-quality labels.
Only five sampled frames are decoded per DICOM. Source files are read-only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dicom_prediction_video import (  # noqa: E402
    VIEWS as MODEL_VIEWS,
    decode_frames,
    doppler_pixels,
    load_view_model,
    normalize,
    normalize_appearance,
    spatial_geometry,
)


VERSION = "echoprime-view-strain-screen-v2"
TARGETS = ("A2C", "A3C", "A4C")
WEIGHTS = Path(r"D:\us\output\dicom_prediction\weights")


def selection_score(probability: float, agreement: float, quality: float) -> float:
    return round(0.75 * probability + 0.10 * agreement + 0.15 * quality, 5)


def quality_metrics(frames: np.ndarray, number_of_frames: int, fps: float) -> dict:
    gray = frames.astype(np.float32).mean(-1)
    foreground = gray > 12
    coverage = float(foreground.mean())
    motion = float(np.abs(np.diff(gray, axis=0)).mean()) if len(frames) > 1 else 0.0
    sharpness = float(np.median([cv2.Laplacian(frame.astype(np.uint8), cv2.CV_32F).var()
                                 for frame in gray]))
    duration = float(number_of_frames / fps) if fps > 0 else 0.0
    # Smooth scores preserve relative differences among otherwise adequate
    # loops; this is a technical screen, not an LV visibility diagnosis.
    duration_score = duration / (duration + 0.8)
    coverage_score = coverage / (coverage + 0.20)
    motion_score = motion / (motion + 5.0)
    sharpness_score = sharpness / (sharpness + 600.0)
    proxy = 0.30 * duration_score + 0.30 * coverage_score + 0.20 * motion_score + 0.20 * sharpness_score
    return {"duration_seconds": round(duration, 3), "foreground_fraction": round(coverage, 4),
            "motion_proxy": round(motion, 3), "sharpness_proxy": round(sharpness, 2),
            "quality_proxy": round(float(proxy), 3)}


def choose_distinct(rows: list[dict]) -> tuple[dict, dict]:
    """Return distinct candidates and the leading review options for each view."""
    options = {}
    for view in TARGETS:
        candidates = []
        for row in rows:
            if row.get("status") != "ok" or not row.get("bmode_candidate"):
                continue
            probability = float(row["probabilities"][MODEL_VIEWS.index(view)])
            agreement = float(row["frame_agreement"][view])
            if probability < 0.15:
                continue
            score = selection_score(probability, agreement, row["quality"]["quality_proxy"])
            candidates.append({"id": row["id"], "probability": round(probability, 4),
                               "agreement": round(agreement, 3), "score": score,
                               "quality_proxy": row["quality"]["quality_proxy"],
                               "predicted_view": row["predicted_view"],
                               "review_level": "strong" if probability >= .75 and agreement >= .6 else "uncertain"})
        options[view] = sorted(candidates, key=lambda item: item["score"], reverse=True)[:5]
    # A blank is preferable to an implausible guess. Distinctness also avoids
    # assigning an ambiguous clip to multiple apical views.
    choices = [[None] + [candidate for candidate in options[view]
                         if candidate["probability"] >= .5 and candidate["agreement"] >= .4]
               for view in TARGETS]
    best_score = float("-inf")
    selected = {view: "" for view in TARGETS}
    for combo in itertools.product(*choices):
        ids = [candidate["id"] for candidate in combo if candidate]
        if len(ids) != len(set(ids)):
            continue
        value = sum((candidate["score"] - .35) for candidate in combo if candidate)
        if value > best_score:
            best_score = value
            selected = {view: candidate["id"] if candidate else ""
                        for view, candidate in zip(TARGETS, combo)}
    # Keep an assigned candidate visible even when avoiding duplicate clips
    # pushes it below the first three options for that view.
    alternatives = {}
    for view in TARGETS:
        alternatives[view] = options[view][:3]
        chosen = selected[view]
        if chosen and all(item["id"] != chosen for item in alternatives[view]):
            alternatives[view].append(next(item for item in options[view] if item["id"] == chosen))
    return selected, alternatives


class Suggester:
    def __init__(self, output: Path, weights: Path = WEIGHTS):
        self.output = output
        self.weights = weights
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_lock = threading.Lock()
        torch.set_num_threads(4)
        cv2.setNumThreads(1)

    def _model(self):
        if self.model is None:
            with self.model_lock:
                if self.model is None:
                    self.model = load_view_model(self.weights, self.device)
        return self.model

    def _fingerprint(self, file: dict) -> str:
        stat = Path(file["path"]).stat()
        text = f"{VERSION}|{file['path']}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, file: dict) -> Path:
        return self.output / "view_predictions" / f"{file['id']}-{self._fingerprint(file)}.json"

    def analyze_file(self, file: dict) -> dict:
        cache = self._cache_path(file)
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        path = file["path"]
        result = {"id": file["id"], "status": "error", "model_version": VERSION}
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            geometry = spatial_geometry(ds)
            if not geometry["eligible_region"]:
                raise ValueError("No declared 2D tissue region; crop requires manual review")
            count = int(ds.NumberOfFrames)
            indices = np.unique(np.linspace(0, count - 1, 5).round().astype(int))
            frames = decode_frames(path, indices, geometry["crop"])
            bmode = not doppler_pixels(frames) and 2 not in geometry["region_types"] and not geometry.get("multiple_tissue_regions", False)
            canonical, appearance = normalize_appearance(frames)
            tensor = torch.from_numpy(canonical).permute(0, 3, 1, 2).float().to(self.device)
            with torch.inference_mode():
                probs = self._model()(normalize(tensor, "echoprime")).softmax(-1).cpu().numpy()
            mean = probs.mean(0)
            frame_labels = [MODEL_VIEWS[int(index)] for index in probs.argmax(1)]
            fps = 0.0
            if ds.get("FrameTime"):
                fps = 1000.0 / float(ds.FrameTime)
            elif ds.get("CineRate"):
                fps = float(ds.CineRate)
            elif ds.get("RecommendedDisplayFrameRate"):
                fps = float(ds.RecommendedDisplayFrameRate)
            else:
                fps = 30.0
            result.update(status="ok", predicted_view=MODEL_VIEWS[int(mean.argmax())],
                          confidence=round(float(mean.max()), 4),
                          probabilities=[round(float(value), 6) for value in mean],
                          frame_agreement={view: round(frame_labels.count(view) / len(frame_labels), 3)
                                           for view in TARGETS},
                          bmode_candidate=bool(bmode),
                          quality=quality_metrics(frames, count, fps),
                          appearance=appearance, crop_source=geometry["crop_source"],
                          warning="Color Doppler or multiple tissue regions" if not bmode else "")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f"{cache.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
            os.replace(temporary, cache)
        finally:
            if temporary.exists():
                temporary.unlink()
        return result

    def analyze_visit(self, visit_key: str, files: list[dict], progress=None) -> dict:
        rows = []
        for index, file in enumerate(files, 1):
            rows.append(self.analyze_file(file))
            if progress:
                progress(index, len(files))
        selected, alternatives = choose_distinct(rows)
        return {"visit_key": visit_key, "status": "ready", "model_version": VERSION,
                "model_name": "EchoPrime ConvNeXt view classifier",
                "selected": selected, "alternatives": alternatives,
                "predictions": {row["id"]: row for row in rows},
                "analyzed": len(rows), "errors": sum(row["status"] != "ok" for row in rows),
                "caveat": "View probabilities and quality proxies are not validated for strain suitability. Review the true apex, endocardial visibility, and a full cardiac cycle before approval."}
