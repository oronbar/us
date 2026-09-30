"""Local, read-only DICOM review server for manual apical-view selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image
import imageio_ffmpeg
import pydicom
from pydicom.pixels import iter_pixels, pixel_array


ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIT = Path(r"D:\us\output\dicom_prediction\ichilov3_20260923")
DEFAULT_OUTPUT = Path(r"D:\us\output\dicom_prediction\ichilov3_manual_selection")
VIEWS = ("A2C", "A3C", "A4C")


def visit_key(patient: str, date: str) -> str:
    return f"{patient}|{date}"


def iso_date(value: str) -> str:
    value = str(value or "")
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return ""


def file_id(path: str) -> str:
    return hashlib.sha256(path.casefold().encode("utf-8")).hexdigest()[:24]


class ReviewData:
    def __init__(self, audit: Path, output: Path):
        self.audit = audit
        self.output = output
        self.lock = threading.Lock()
        self.clip_lock = threading.Lock()
        self.suggestion_lock = threading.Lock()
        self.suggestion_compute_lock = threading.Lock()
        self.suggestion_jobs = {}
        self.suggester = None
        self.visits = {}
        self.files = {}
        self.files_by_visit = {}
        self.mismatched_folder_dates = {}
        self.decisions = {}
        self._load_audit()
        self._load_decisions()

    def _load_audit(self):
        with (self.audit / "visits.json").open(encoding="utf-8") as handle:
            for row in json.load(handle):
                key = visit_key(row["patient"], row["date"])
                self.visits[key] = {
                    "key": key,
                    "patient": row["patient"],
                    "date": row["date"],
                    "strain_report": row["strain_report"],
                    "matching_dicom_folder": row["matching_dicom_folder"],
                    "tomtec_bookmark": row["tomtec_bookmark"],
                    "resolved_views": row["resolved_views"],
                    "audit_views": {view: row.get(view, "") for view in VIEWS},
                    "audit_notes": row.get("notes", ""),
                    "dicom_folders": row.get("dicom_folders", ""),
                    "session_selection": row.get("session_selection", ""),
                }
        with (self.audit / "inventory.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") != "ok":
                    continue
                try:
                    frames = int(record.get("NumberOfFrames") or 0)
                except ValueError:
                    continue
                if frames < 2:
                    continue
                patient = record.get("patient_folder", "")
                date = iso_date(record.get("StudyDate", ""))
                key = visit_key(patient, date)
                folder_date = iso_date(record.get("date_folder", "").replace("_", ""))
                folder_key = visit_key(patient, folder_date)
                if folder_date and date and folder_date != date and folder_key in self.visits:
                    self.mismatched_folder_dates.setdefault(folder_key, set()).add(date)
                if key not in self.visits:
                    continue
                path = record["path"]
                identity = file_id(path)
                try:
                    instance = int(record.get("InstanceNumber") or 0)
                except ValueError:
                    instance = 0
                item = {
                    "id": identity,
                    "visit_key": key,
                    "path": path,
                    "name": Path(path).name,
                    "folder_date": record.get("date_folder", ""),
                    "study_date": date,
                    "study_uid": record.get("StudyInstanceUID", ""),
                    "series_uid": record.get("SeriesInstanceUID", ""),
                    "sop_uid": record.get("SOPInstanceUID", ""),
                    "frames": frames,
                    "rows": record.get("Rows", ""),
                    "columns": record.get("Columns", ""),
                    "instance": instance,
                    "series_description": record.get("SeriesDescription", ""),
                    "image_comments": record.get("ImageComments", ""),
                    "image_type": record.get("ImageType", ""),
                    "manufacturer": record.get("Manufacturer", ""),
                    "content_time": record.get("ContentTime", ""),
                    "size": record.get("size", 0),
                }
                self.files[identity] = item
                self.files_by_visit.setdefault(key, []).append(identity)
        for ids in self.files_by_visit.values():
            ids.sort(key=lambda identity: (self.files[identity]["instance"], self.files[identity]["path"]))

    def _load_decisions(self):
        path = self.output / "decisions.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.decisions = loaded.get("decisions", {})

    def visit_list(self):
        result = []
        for key, item in self.visits.items():
            decision = self.decisions.get(key, {})
            result.append({**item, "video_count": len(self.files_by_visit.get(key, [])),
                           "manual_status": decision.get("status", "unreviewed"),
                           "manual_updated_at": decision.get("updated_at", "")})
        return sorted(result, key=lambda row: (row["patient"], row["date"]))

    def visit_detail(self, key: str):
        if key not in self.visits:
            raise KeyError("Visit not found")
        return {"visit": self.visits[key],
                "files": [self.files[identity] for identity in self.files_by_visit.get(key, [])],
                "folder_contains_study_dates": sorted(self.mismatched_folder_dates.get(key, set())),
                "decision": self.decisions.get(key)}

    def save(self, payload: dict):
        key = str(payload.get("visit_key", ""))
        if key not in self.visits:
            raise ValueError("Unknown visit")
        views = payload.get("views")
        if not isinstance(views, dict) or set(views) != set(VIEWS):
            raise ValueError("Provide A2C, A3C, and A4C selections")
        if self.visits[key]["resolved_views"] == 3 and self.visits[key]["tomtec_bookmark"] == "Yes":
            raise ValueError("This visit already has three resolved TOMTEC clips")
        selected = [str(views[view]) for view in VIEWS if views[view]]
        if len(selected) != len(set(selected)):
            raise ValueError("Each view must use a different DICOM file")
        valid = set(self.files_by_visit.get(key, []))
        if any(identity not in valid for identity in selected):
            raise ValueError("Selections must be cine DICOMs from this visit's DICOM study date")
        sources = payload.get("selection_source") or {}
        if not isinstance(sources, dict) or any(sources.get(view, "") not in ("", "manual", "suggested", "tomtec") for view in VIEWS):
            raise ValueError("Invalid selection provenance")
        for view in VIEWS:
            if sources.get(view) == "tomtec" and views[view]:
                if self.visits[key]["tomtec_bookmark"] != "Yes" or self.files[str(views[view])]["path"] != self.visits[key]["audit_views"][view]:
                    raise ValueError("TOMTEC provenance must match the recovered bookmark clip")
        reviewer = str(payload.get("reviewer", "")).strip()[:100]
        note = str(payload.get("note", "")).strip()[:2000]
        unable = payload.get("unable") is True  # Existing records/API clients.
        invalid = payload.get("invalid") is True
        if not reviewer:
            raise ValueError("Reviewer name is required")
        if unable and not note:
            raise ValueError("Add a reason when marking a visit unable to resolve")
        if unable and selected:
            raise ValueError("Clear the selections before marking unable to resolve")
        if invalid and selected:
            raise ValueError("Clear the selections before marking the visit invalid")
        if invalid and not note:
            note = "No appropriate three-view cine combination found."
        status = "invalid" if invalid else "unable" if unable else ("complete" if len(selected) == 3 else "partial")
        entry = {
            "visit_key": key,
            "patient": self.visits[key]["patient"],
            "date": self.visits[key]["date"],
            "views": {view: str(views[view]) if views[view] else "" for view in VIEWS},
            "selection_source": {view: sources.get(view, "") if views[view] else "" for view in VIEWS},
            "reviewer": reviewer,
            "note": note,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        with self.lock:
            next_decisions = {**self.decisions, key: entry}
            target = self.output / "decisions.json"
            if target.exists():
                backup = self.output / f"decisions.backup.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
                shutil.copy2(target, backup)
            temp = self.output / f"decisions.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with temp.open("w", encoding="utf-8") as handle:
                    json.dump({"schema_version": 1, "decisions": next_decisions}, handle,
                              ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
            finally:
                if temp.exists():
                    temp.unlink()
            self.decisions = next_decisions
        return entry

    def export_rows(self):
        rows = []
        for key, visit in self.visits.items():
            decision = self.decisions.get(key)
            if not decision:
                continue
            row = {"patient": visit["patient"], "visit_date": visit["date"],
                   "status": decision["status"], "reviewer": decision["reviewer"],
                   "updated_at": decision["updated_at"], "note": decision["note"],
                   "strain_report": visit["strain_report"],
                   "tomtec_bookmark": visit["tomtec_bookmark"]}
            for view in VIEWS:
                identity = decision["views"].get(view, "")
                chosen = self.files.get(identity, {})
                row[f"{view}_path"] = chosen.get("path", "")
                row[f"{view}_SOPInstanceUID"] = chosen.get("sop_uid", "")
                row[f"{view}_StudyInstanceUID"] = chosen.get("study_uid", "")
                row[f"{view}_selection_source"] = decision.get("selection_source", {}).get(view, "")
            rows.append(row)
        return sorted(rows, key=lambda row: (row["patient"], row["visit_date"]))

    def clip_path(self, item: dict) -> Path:
        source = Path(item["path"])
        stat = source.stat()
        cache = self.output / "cine_cache"
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"{item['id']}-{stat.st_size}-{stat.st_mtime_ns}.mp4"
        if target.is_file():
            return target
        with self.clip_lock:
            if target.is_file():
                return target
            encode_clip(source, target)
        return target

    def _suggestion_signature(self, key: str) -> str:
        parts = ["echoprime-view-strain-screen-v2", key]
        weights = Path(r"D:\us\output\dicom_prediction\weights\view_classifier.pt")
        stat = weights.stat()
        parts.append(f"weights:{stat.st_size}:{stat.st_mtime_ns}")
        for identity in self.files_by_visit.get(key, []):
            item = self.files[identity]
            stat = Path(item["path"]).stat()
            parts.append(f"{identity}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]

    def _suggestion_path(self, key: str) -> Path:
        token = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return self.output / "suggestions" / f"{token}.json"

    def suggestions(self, key: str, start: bool = False) -> dict:
        if key not in self.visits:
            raise KeyError("Visit not found")
        if not self.files_by_visit.get(key):
            return {"status": "no_cines", "visit_key": key}
        signature = self._suggestion_signature(key)
        path = self._suggestion_path(key)
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("source_signature") == signature:
                    return cached
            except (OSError, ValueError):
                pass
        with self.suggestion_lock:
            job = self.suggestion_jobs.get(key)
            if job and job.get("source_signature") == signature and job["status"] in ("queued", "running"):
                return dict(job)
            if not start:
                return {"status": "not_started", "visit_key": key}
            job = {"status": "queued", "visit_key": key, "source_signature": signature,
                   "done": 0, "total": len(self.files_by_visit[key])}
            self.suggestion_jobs[key] = job
        thread = threading.Thread(target=self._run_suggestions, args=(key, signature), daemon=True)
        thread.start()
        return dict(job)

    def _run_suggestions(self, key: str, signature: str):
        try:
            with self.suggestion_compute_lock:
                with self.suggestion_lock:
                    self.suggestion_jobs[key]["status"] = "running"
                if self.suggester is None:
                    from suggestions import Suggester
                    self.suggester = Suggester(self.output)
                files = [self.files[identity] for identity in self.files_by_visit[key]]

                def progress(done, total):
                    with self.suggestion_lock:
                        self.suggestion_jobs[key].update(done=done, total=total)

                result = self.suggester.analyze_visit(key, files, progress=progress)
                result["source_signature"] = signature
                path = self._suggestion_path(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    backup = path.with_name(f"{path.stem}.backup.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")
                    shutil.copy2(path, backup)
                temp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
                try:
                    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(temp, path)
                finally:
                    if temp.exists():
                        temp.unlink()
                with self.suggestion_lock:
                    self.suggestion_jobs[key] = {"status": "ready", "visit_key": key, "source_signature": signature,
                                                 "done": len(files), "total": len(files)}
        except Exception as exc:
            with self.suggestion_lock:
                self.suggestion_jobs[key] = {"status": "error", "visit_key": key,
                                             "source_signature": signature,
                                             "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def cine_fps(dataset) -> float:
    try:
        frame_time = float(dataset.get("FrameTime", 0) or 0)
        if frame_time > 0:
            return max(5.0, min(120.0, 1000.0 / frame_time))
    except (TypeError, ValueError):
        pass
    for key in ("CineRate", "RecommendedDisplayFrameRate"):
        try:
            value = float(dataset.get(key, 0) or 0)
            if value > 0:
                return max(5.0, min(120.0, value))
        except (TypeError, ValueError):
            pass
    return 30.0


def rgb_image(array: np.ndarray, max_width: int | None = None) -> Image.Image:
    if array.dtype != np.uint8:
        high = float(np.max(array)) or 1.0
        array = np.asarray(np.clip(array.astype(np.float32) * (255.0 / high), 0, 255), dtype=np.uint8)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[:, :, :3]
    image = Image.fromarray(array).convert("RGB")
    if max_width and image.width > max_width:
        width = max_width - max_width % 2
        height = max(2, round(image.height * width / image.width / 2) * 2)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def encode_clip(source: Path, target: Path) -> None:
    """Create an H.264 preview in output storage; never writes to the DICOM path."""
    metadata = pydicom.dcmread(str(source), stop_before_pixels=True)
    frames = iter_pixels(str(source))
    try:
        first = rgb_image(next(frames), 900)
    except StopIteration as exc:
        raise ValueError("DICOM contains no decodable frames") from exc
    width, height = first.size
    fps = cine_fps(metadata)
    temp = target.with_name(f"{target.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
               "-framerate", f"{fps:.3f}", "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-y", str(temp)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        process.stdin.write(first.tobytes())
        for frame in frames:
            image = rgb_image(frame, 900)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0 or not temp.is_file() or temp.stat().st_size == 0:
            raise RuntimeError(f"Video encoding failed: {error[-500:]}")
        os.replace(temp, target)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        if temp.exists():
            temp.unlink()


def jpeg_frame(item: dict, index: int, max_width: int) -> bytes:
    path = Path(item["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 0 <= index < item["frames"]:
        raise ValueError("Frame outside this cine loop")
    image = rgb_image(pixel_array(str(path), index=index), max_width)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=78)
    return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def _send(self, body: bytes, content_type: str, status: int = 200, extra=None,
              cache_control: str = "no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status=200):
        self._send(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def _error(self, status: int, message: str):
        self._json({"error": message}, status)

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/app.js", "/style.css", "/viewer.css", "/suggestions.css"):
                name = "index.html" if parsed.path == "/" else parsed.path[1:]
                content_type = {"index.html": "text/html; charset=utf-8",
                                "app.js": "text/javascript; charset=utf-8",
                                "style.css": "text/css; charset=utf-8",
                                "viewer.css": "text/css; charset=utf-8",
                                "suggestions.css": "text/css; charset=utf-8"}[name]
                return self._send((ROOT / name).read_bytes(), content_type)
            if parsed.path == "/api/visits":
                return self._json(self.server.data.visit_list())
            if parsed.path == "/api/visit":
                return self._json(self.server.data.visit_detail(query.get("key", [""])[0]))
            if parsed.path == "/api/suggestions":
                key = query.get("key", [""])[0]
                return self._json(self.server.data.suggestions(key, query.get("start", ["0"])[0] == "1"))
            if parsed.path == "/api/frame":
                identity = query.get("id", [""])[0]
                item = self.server.data.files.get(identity)
                if not item:
                    return self._error(404, "Unknown DICOM")
                index = int(query.get("index", [str(item["frames"] // 2)])[0])
                width = min(900, max(200, int(query.get("width", ["640"])[0])))
                return self._send(jpeg_frame(item, index, width), "image/jpeg",
                                  cache_control="private, max-age=3600")
            if parsed.path == "/api/clip":
                identity = query.get("id", [""])[0]
                item = self.server.data.files.get(identity)
                if not item:
                    return self._error(404, "Unknown DICOM")
                clip = self.server.data.clip_path(item)
                return self._send(clip.read_bytes(), "video/mp4",
                                  cache_control="private, max-age=3600")
            if parsed.path == "/api/export.csv":
                rows = self.server.data.export_rows()
                output = io.StringIO()
                fields = ["patient", "visit_date", "status", "reviewer", "updated_at", "note",
                          "strain_report", "tomtec_bookmark"]
                for view in VIEWS:
                    fields += [f"{view}_path", f"{view}_SOPInstanceUID", f"{view}_StudyInstanceUID", f"{view}_selection_source"]
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                return self._send(('\ufeff' + output.getvalue()).encode("utf-8"), "text/csv; charset=utf-8",
                                  extra={"Content-Disposition": 'attachment; filename="ichilov3_manual_views.csv"'})
            return self._error(404, "Not found")
        except KeyError as exc:
            self._error(404, str(exc))
        except (ValueError, IndexError) as exc:
            self._error(400, str(exc))
        except (OSError, RuntimeError) as exc:
            self._error(500, f"Unable to load DICOM: {exc}")

    def do_POST(self):
        if self.path != "/api/decision":
            return self._error(404, "Not found")
        if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}",
                                               f"http://localhost:{self.server.server_port}"):
            return self._error(403, "Invalid origin")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 10000:
                return self._error(413, "Invalid request size")
            payload = json.loads(self.rfile.read(size))
            return self._json(self.server.data.save(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error(400, str(exc))
        except OSError as exc:
            return self._error(500, f"Unable to save decision: {exc}")

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/frame"):
            super().log_message(fmt, *args)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, data):
        super().__init__(address, Handler)
        self.data = data
        self.daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    data = ReviewData(args.audit, args.output)
    server = ReviewServer(("127.0.0.1", args.port), data)
    print(f"Review app: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Visits: {len(data.visits)}; cine files: {len(data.files)}; decisions: {len(data.decisions)}", flush=True)
    print(f"Decisions: {args.output / 'decisions.json'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
