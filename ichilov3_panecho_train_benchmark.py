"""Short, disposable GPU timing/memory benchmark for PanEcho adaptation."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dicom_prediction_encode import load_encoder


OUT = Path(r"D:\DS\ichilov3_rwave_first_cycle_panecho_20261005\train_benchmark.json")
ART = Path(r"D:\us\output\dicom_prediction")


def trial(model, mode, steps=4, batch=12):
    model.requires_grad_(False)
    model.transformer.requires_grad_(True)
    if mode == "temporal_plus_last_convnext":
        model.encoder.model.features[-1].requires_grad_(True)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
    x = torch.rand((batch, 3, 16, 224, 224), device="cuda")
    target = torch.randn((batch, 768), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for _ in range(steps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = F.mse_loss(pred.float(), target)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        return {"status": "ok", "seconds_per_step": times,
                "steady_median_seconds": sorted(times[1:])[len(times[1:]) // 2],
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "error": str(exc)[:250]}
    finally:
        del x, target, opt
        torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4)
    model = load_encoder("panecho", ART, "cuda")
    result = {"gpu": torch.cuda.get_device_name(), "batch": 12,
              "frames_per_clip": 16, "resolution": 224,
              "temporal_only": trial(model, "temporal_only")}
    torch.cuda.empty_cache()
    result["temporal_plus_last_convnext"] = trial(model, "temporal_plus_last_convnext")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
