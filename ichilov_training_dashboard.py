"""Read-only live dashboard for Ichilov video adaptation experiments.

Only aggregate progress and metrics are exposed. Source DICOMs and patient-level
evaluation files are never served. Binds to loopback and the Tailscale IPv4 address.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


ROOT = Path(r"D:\DS\ichilov_ge_ssl_20260930")
PANECHO_ROOT = Path(r"D:\DS\ichilov3_panecho_rwave_ssl_20261005")
EXPERIMENT = Path(r"D:\us\output\ichilov_video_experiments_20260930")
INDEX = Path(__file__).with_name("ichilov_training_dashboard.html")
KEY = re.compile(r"^(all|A2C|A3C|A4C)_r([0-2])_f([0-4])$")
LOSS = re.compile(r"SSL (all|A2C|A3C|A4C) r(\d+) f(\d+): (\d+)/(\d+) loss=([\d.]+) \((\d+)s\)")
PANECHO_KEY = re.compile(r"^r([0-2])_f([0-4])$")
PANECHO_LOSS = re.compile(r"PanEcho SSL r\d+ f\d+ (\d+)/(\d+) loss=([\d.]+) elapsed=(\d+)s")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def read_loss(key: str) -> list[dict]:
    if not KEY.fullmatch(key):
        return []
    path = ROOT / "logs" / f"train_{key}.log"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    points = []
    for line in lines:
        match = LOSS.search(line)
        if match:
            points.append({"step": int(match.group(4)), "total": int(match.group(5)),
                           "loss": float(match.group(6)), "seconds": int(match.group(7))})
    return points


def panecho_loss(key: str) -> list[dict]:
    if not PANECHO_KEY.fullmatch(key):
        return []
    path = PANECHO_ROOT / "logs" / f"train_{key}.log"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    points = []
    for line in lines:
        match = PANECHO_LOSS.search(line)
        if match:
            points.append({"step": int(match.group(1)), "total": int(match.group(2)),
                           "loss": float(match.group(3)), "seconds": int(match.group(4))})
    return points


def panecho_status() -> dict:
    runner = read_json(PANECHO_ROOT / "runner_status.json") or {}
    models = []
    for repeat in range(3):
        for fold in range(5):
            key = f"r{repeat}_f{fold}"
            training = PANECHO_ROOT / "training" / key
            encoding = PANECHO_ROOT / "evaluation_embeddings" / key
            status = read_json(training / "status.json") or {}
            complete = read_json(training / "run_complete.json")
            encoded = read_json(encoding / "run_complete.json")
            models.append({"key": key, "repeat": repeat + 1, "fold": fold + 1,
                           "step": status.get("step", 0),
                           "total": status.get("total", 0),
                           "loss": status.get("loss"),
                           "train_complete": bool(complete),
                           "encoded": bool(encoded),
                           "minutes": round(complete["elapsed_seconds"] / 60, 1)
                           if complete else None})
    result = read_json(PANECHO_ROOT / "evaluation" / "results.json") or {}
    return {"updated_at": datetime.now(timezone.utc).isoformat(),
            "runner": runner.get("stage", "unknown"),
            "active_key": re.search(r"r[0-2]_f[0-4]", runner.get("stage", ""))
            .group(0) if re.search(r"r[0-2]_f[0-4]", runner.get("stage", "")) else None,
            "progress": {"trained": sum(m["train_complete"] for m in models),
                         "encoded": sum(m["encoded"] for m in models)},
            "models": models,
            "evaluation_complete": (PANECHO_ROOT / "evaluation" /
                                    "run_complete.json").exists(),
            "complete": (PANECHO_ROOT / "run_complete.json").exists(),
            "metrics": {"gls": result.get("same_visit_mid_gls"),
                        "future": (result.get("next_visit") or {}).get("metrics")},
            "gpu": gpu()}


def model_summary(view: str, repeat: int, fold: int) -> dict:
    key = f"{view}_r{repeat}_f{fold}"
    complete = read_json(ROOT / view / f"r{repeat}_f{fold}" / "run_complete.json")
    log = ROOT / "logs" / f"train_{key}.log"
    latest = None
    try:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 1024))
            tail = handle.read().decode("utf-8", errors="replace")
        matches = list(LOSS.finditer(tail))
        if matches:
            latest = matches[-1]
    except OSError:
        pass
    return {
        "key": key, "view": view, "repeat": repeat + 1, "fold": fold + 1,
        "complete": bool(complete),
        "step": int(latest.group(4)) if latest else 0,
        "total": int(latest.group(5)) if latest else 0,
        "loss": float(latest.group(6)) if latest else None,
        "minutes": round(float(complete["elapsed_seconds"]) / 60, 1) if complete else None,
    }


def gpu() -> dict | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=3, check=True,
        )
        fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        return {"name": fields[0], "utilization": int(fields[1]),
                "memory_used_mb": int(fields[2]), "memory_total_mb": int(fields[3]),
                "temperature_c": int(fields[4])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def metrics() -> dict:
    stage1 = read_json(EXPERIMENT / "stage1" / "checkpoint_comparison.json") or {}
    stage2 = read_json(EXPERIMENT / "stage2" / "checkpoint_comparison.json") or {}
    out = {"reference": stage1.get("reference"),
           "frozen": {key: stage1.get(key) for key in ("echofm", "echoprime", "panecho")},
           "frozen_gls": stage1.get("gls_pearson_r"),
           "history": stage2.get("history_vs_current")}
    for config in ("all", "specific"):
        data = read_json(ROOT / "evaluation" / config / "checkpoint_comparison.json")
        if data:
            out[config] = {
                "current": next((row for row in data.get("future", [])
                                 if row.get("mode") == "current" and row.get("arm") == "late25"), None),
                "history": next((row for row in data.get("future", [])
                                 if row.get("mode") == "history" and row.get("arm") == "late25"), None),
                "gls": data.get("gls", [None])[0],
            }
    return out


def status_payload() -> dict:
    runner = read_json(ROOT / "runner_status.json") or {}
    active = runner.get("job", "")
    models = [model_summary(view, repeat, fold)
              for view in ("all", "A2C", "A3C", "A4C")
              for repeat in range(3) for fold in range(5)]
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runner": {"status": runner.get("status", "unknown"), "job": active,
                   "next_action": runner.get("next_action")},
        "active_key": active.removeprefix("train_") if active.startswith("train_") else None,
        "progress": {"shared": sum(m["complete"] for m in models if m["view"] == "all"),
                     "specific": sum(m["complete"] for m in models if m["view"] != "all")},
        "models": models, "metrics": metrics(), "gpu": gpu(),
        "complete": (ROOT / "run_complete.json").exists(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/":
            try:
                body = INDEX.read_bytes()
            except OSError:
                self.send_error(500, "Dashboard file unavailable")
                return
            self.reply(body, "text/html; charset=utf-8")
        elif url.path == "/api/status":
            self.reply(json.dumps(status_payload(), allow_nan=False).encode(), "application/json")
        elif url.path == "/api/loss":
            key = parse_qs(url.query).get("key", [""])[0]
            if not KEY.fullmatch(key):
                self.send_error(400, "Invalid model key")
                return
            self.reply(json.dumps({"key": key, "points": read_loss(key)}).encode(), "application/json")
        elif url.path == "/api/panecho/status":
            self.reply(json.dumps(panecho_status(), allow_nan=False).encode(), "application/json")
        elif url.path == "/api/panecho/loss":
            key = parse_qs(url.query).get("key", [""])[0]
            if not PANECHO_KEY.fullmatch(key):
                self.send_error(400, "Invalid PanEcho fold key")
                return
            self.reply(json.dumps({"key": key, "points": panecho_loss(key)}).encode(),
                       "application/json")
        else:
            self.send_error(404)

    def reply(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Ichilov dashboard: http://{host}:{port}/", flush=True)
    server.serve_forever(poll_interval=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--tailscale-ip", default="100.73.61.63")
    args = parser.parse_args()
    hosts = ["127.0.0.1"]
    if args.tailscale_ip and args.tailscale_ip not in hosts:
        hosts.append(args.tailscale_ip)
    threads = [threading.Thread(target=serve, args=(host, args.port), daemon=True)
               for host in hosts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
