# Ichilov video experiment run control

The authorized three-stage run is tracked at
`D:\us\output\ichilov_video_experiments_20260930\state.json`; its detailed
protocol is in the adjacent `PLAN.md`. Patient-level outputs under that
directory and under `D:\DS\ichilov_*_20260930` are private and untracked.

1. `ichilov3_encode_echofm.py` writes each selected cine atomically, then the
   411-visit array under `D:\DS\ichilov3_echofm_20260930`.
2. `ichilov_video_stage12_runner.py` waits for full encoding, then runs the
   patient-held-out Stage 1 and 2 probes, resuming completed folds.
3. `ichilov_ge_stage3_prepare_runner.py` inventories all three original DICOM
   datasets read-only, deduplicates GE cines, and applies the released
   EchoPrime view classifier where needed.
4. `ichilov_ge_ssl_runner.py` waits for classification, caches temporal cine
   windows, verifies a frozen EchoPrime prefix cache, trains 15 fold-specific
   shared adapters and evaluates them, then trains 45 independent view-specific
   adapters and evaluates them. Each adapter sees every eligible cine from its
   fold's training patients once and checkpoints every 25 steps. A held-out
   patient's DICOMs are excluded from that fold's unlabeled training data.

The two Stage 3 runners write `prepare_status.json` and `runner_status.json`
under their respective `D:\DS` folders. Before restarting one, verify that
no matching Python process is active, that its latest log and checkpoint are
consistent, and that the `gpu_job.lock` owner is not running. Never launch a
second GPU job while another encoder, classifier, prefix cache, adapter train,
or adapter encode process is active. Original DICOMs and reports are never
modified.

A 30-minute thread heartbeat named `Continue Ichilov video experiments`
continues the run if a process stops, stays quiet during normal progress,
and must be paused after all evaluations complete.
