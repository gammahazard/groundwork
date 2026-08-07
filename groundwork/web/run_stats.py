"""Per-run training stats + peek artifacts + GPU sample — pure READERS of
what ultralytics writes into a run dir (results.csv, args.yaml, batch/label
images). Shared by the live dashboard progress (retrain_job._progress) and
the Runs-tab post-mortem 🔬 (api_history /api/runs/{run}/peek). Split out of
retrain_job so that module stays orchestration-only (state, single-flight,
worker, cancel)."""
from __future__ import annotations
import os
import shutil
import subprocess

from ..dataset import paths
from ..dataset.pipeline import training_history


def _outputs_url(p) -> str:
    """A filesystem path under outputs/, as the URL the /outputs mount serves.

    apps/webui.py mounts OUTPUTS_DIR at /outputs, so the browser-visible URL for
    any artifact is just its path relative to that root. Deriving it keeps ONE
    statement of where a run lives; spelling it out by hand is how this file came
    to disagree with paths.RUNS_DIR for a project whose dataset_root moves.
    """
    from ..config import OUTPUTS_DIR
    try:
        return "/outputs/" + str(p.resolve().relative_to(OUTPUTS_DIR.resolve()))
    except ValueError:
        # Outside outputs/ entirely — nothing the mount can serve. Say so rather
        # than emit a URL that 404s.
        return ""


def stats(run) -> dict:
    """Epochs, curves and peek artifacts for ONE run dir. `times` (cumulative
    seconds per epoch) is included for pace math; strip it for API payloads."""
    total = close_mosaic = None
    args = run / "args.yaml"
    if args.exists():
        for ln in args.read_text(errors="ignore").splitlines():
            for key in ("epochs:", "close_mosaic:"):
                if ln.strip().startswith(key):
                    try:
                        v = int(float(ln.split(":", 1)[1]))
                        total = v if key == "epochs:" else total
                        close_mosaic = v if key == "close_mosaic:" else close_mosaic
                    except ValueError:
                        pass
    out = {"run": run.name, "epoch": 0, "total": total, "close_mosaic": close_mosaic}
    # Peek artifacts ultralytics already wrote (served via the /outputs mount):
    # train_batch0-2 = actual augmented (mosaic) batches from training start;
    # high-numbered ones appear when mosaic closes = real un-collaged layouts.
    nums = []
    for p in run.glob("train_batch*.jpg"):
        try:
            nums.append((int(p.stem.replace("train_batch", "")), p.name))
        except ValueError:
            pass
    nums.sort()
    out["peek"] = {
        # DERIVED, not spelled. This was the literal `/outputs/dataset/runs/...`,
        # a fourth independent statement of where runs live (after paths.RUNS_DIR,
        # runtime_model.RUNS and yolo_counter.RUNS_DIR) and the nastiest of the
        # four: it is a URL for the /outputs StaticFiles mount, so a project whose
        # dataset_root is elsewhere would return 404s for every peek image with
        # nothing in any log to say why.
        "base": _outputs_url(run),
        "mosaic_batches": [n for i, n in nums if i < 100],
        "real_batches": [n for i, n in nums if i >= 100],
        "labels": (run / "labels.jpg").exists(),
        "val_pairs": sorted(p.name for p in run.glob("val_batch*_pred.jpg")),
    }
    # Live batch stream (LiveVizTrainer's rolling every-N-steps snapshots).
    out["peek"]["stream"] = sorted(
        p.name for p in (run / "live").glob("batch_step*.jpg")) \
        if (run / "live").exists() else []
    # "Watch it learn" filmstrip + "where it looks" heatmaps, if the post-run
    # generators ran (both under <run>/live/, served via the /outputs mount).
    import json as _json
    for key, fname in (("timelapse", "timelapse.json"),
                       ("heatmaps", "heatmaps.json")):
        f = run / "live" / fname
        if f.exists():
            try:
                out["peek"][key] = _json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    csv = run / "results.csv"
    if not csv.exists():
        return out
    rows = [r for r in csv.read_text(errors="ignore").splitlines() if r.strip()]
    if len(rows) < 2:
        return out
    head = [h.strip() for h in rows[0].split(",")]
    cols = {k: (head.index(k) if k in head else None) for k in
            ("time", "metrics/mAP50(B)", "train/box_loss",
             "train/cls_loss", "train/dfl_loss")}
    series = {k: [] for k in cols}
    for r in rows[1:]:
        parts = r.split(",")
        for k, i in cols.items():
            if i is None:
                continue
            try:
                v = float(parts[i])
                series[k].append(v if k == "time" else round(v, 4))
            except (ValueError, IndexError):
                pass
    maps = series["metrics/mAP50(B)"]
    out.update(epoch=len(rows) - 1, map50=maps[-1] if maps else None,
               map50_curve=maps, box_loss_curve=series["train/box_loss"],
               cls_loss_curve=series["train/cls_loss"],
               dfl_loss_curve=series["train/dfl_loss"], times=series["time"])
    return out


def peek(name: str, pp) -> dict | None:
    """Post-mortem peek payload for a FINISHED run (Runs-tab 🔬)."""
    run = pp.RUNS_DIR / name
    if not run.is_dir():
        return None
    out = stats(run)
    out.pop("times", None)
    return out


def prev_completed(pre: set, pp):
    """The newest completed run that predates the current retrain — the
    baseline the live trajectory is compared against."""
    cands = [p for p in pp.RUNS_DIR.glob("*")
             if p.is_dir() and p.name in pre and (p / "results.csv").exists()
             and (p / "weights" / "best.pt").exists()]
    return max(cands, key=lambda p: (p / "weights" / "best.pt").stat().st_mtime) \
        if cands else None


def eval_overhead_s() -> int:
    """Everything a retrain does besides the epochs: split, scoring BOTH
    checkpoints' conf×iou sweep on the holdout, previews. Estimated from the
    last recorded run's (total duration - training time) — measured on THIS
    machine and holdout size, so the dashboard ETA means 'until done', not
    'until the epochs end'.

    "On THIS machine" is now enforced rather than assumed. Adopted runs used to
    have no duration at all, so they self-excluded; they carry the lab's real
    wall-clock now, and the lab's GPU is not this one. Rows with no machine
    predate the field and were all local."""
    from ..config import MACHINE
    try:
        for rec in reversed(training_history.load()):
            if rec.get("machine") not in (None, MACHINE):
                continue
            d, t = rec.get("duration_s"), rec.get("train_time_s")
            if d and t and d > t:
                return int(d - t)
    except Exception:  # noqa: BLE001
        pass
    return 180                      # placeholder until a first run is recorded


# Once a sample has wedged, this process never asks the driver again. A wedged
# nvidia-smi is uninterruptible, so the leak is permanent and the SECOND attempt
# is what turns one stuck process into forty. Deliberately module-level and
# never reset: "the driver is unhappy" does not become false while we watch.
_SMI_WEDGED = False


def gpu(timeout: float = 2.0) -> dict | None:
    """One nvidia-smi sample. NEVER call this from a request handler.

    Use `gpu_facts` for anything a browser polls — it reads the training log
    instead, and cannot block. This exists for one-shot CLI and setup probes.

    THREE THINGS HERE ARE LOAD-BEARING, all of them scars:

    1. `subprocess.run(timeout=)` does NOT bound this call. On timeout it calls
       kill() then communicate(), and a D-state process cannot be killed, so
       communicate() blocks in wait4() forever. So we drive Popen ourselves and
       simply WALK AWAY from a child that overran — leaked, but not blocking.
    2. ONE executable, not a loop over two. The old version tried `nvidia-smi`
       then the absolute WSL path, so a single wedged call leaked two processes.
       The absolute path is tried only when the bare name is genuinely absent.
    3. A wedge trips `_SMI_WEDGED` for the life of the process (see above).
    """
    global _SMI_WEDGED
    if _SMI_WEDGED:
        return None
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not os.path.exists(exe):
        return None
    try:
        p = subprocess.Popen(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return None
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Do NOT kill-then-wait. Under WSL2 this child may be unkillable, and
        # waiting on it is the bug this whole comment block exists about. Mark
        # the driver bad, abandon the child, return.
        _SMI_WEDGED = True
        return None
    except Exception:  # noqa: BLE001
        return None
    try:
        u, mu, mt = [x.strip() for x in out.strip().split(",")[:3]]
        return {"util": int(u), "vram_used_gb": round(int(mu) / 1024, 1),
                "vram_total_gb": round(int(mt) / 1024, 1)}
    except (ValueError, IndexError):
        return None
