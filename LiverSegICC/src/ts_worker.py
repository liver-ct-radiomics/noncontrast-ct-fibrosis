"""Runs TotalSegmentator for one CT. Started with the Python that HAS TotalSegmentator, which may
be a different installation from the one running the program, so this file is self-contained:
it needs only numpy, SimpleITK and totalsegmentator.

  python ts_worker.py --probe
      prints  INFO|{json}  describing the installation
  python ts_worker.py --ct CT --out MASK --info JSON --tmp DIR [--device auto|gpu|cpu]
      prints  PROG|<percent>|<message>, then DONE|<seconds> or FAIL|<message>
      exit code: 0 done, 3 no liver found, 1 any other error

Path: orient to RAS -> NIfTI -> TotalSegmentator (task=total, roi_subset=[liver], full 1.5 mm
model) -> nearest-neighbour resampling back onto the CT grid.
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# The program's other modules must not shadow names that third-party packages import.
_here = str(Path(__file__).resolve().parent).lower()
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()).lower() != _here]
os.environ.setdefault("MPLBACKEND", "Agg")

TASK = "total"
ORGAN = "liver"
MIN_GPU_GB = 4.0   # below this the full model is run on the CPU


def out(*parts):
    print("|".join(str(p) for p in parts), flush=True)


def probe():
    info = {"ok": False}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from importlib.metadata import version
            info["version"] = version("TotalSegmentator")
            import numpy
            import SimpleITK
            import torch
            info["torch"] = torch.__version__
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
            import totalsegmentator.python_api
            from totalsegmentator.map_to_binary import class_map
            if ORGAN not in class_map[TASK].values():
                raise RuntimeError("this version has no liver class")
            try:
                from totalsegmentator.config import get_weights_dir
                info["weights_ready"] = any(Path(get_weights_dir()).glob("Dataset291_*"))
            except Exception:
                info["weights_ready"] = None
            info["ok"] = True
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    out("INFO", json.dumps(info))
    return 0


def _read(path):
    import SimpleITK as sitk
    if str(path).isascii():
        return sitk.ReadImage(str(path))
    previous = os.getcwd()
    os.chdir(path.parent)
    try:
        return sitk.ReadImage(path.name)
    finally:
        os.chdir(previous)


def _write(image, path, compress = True):
    import SimpleITK as sitk
    if str(path).isascii():
        sitk.WriteImage(image, str(path), compress)
        return
    previous = os.getcwd()
    os.chdir(path.parent)
    try:
        sitk.WriteImage(image, path.name, compress)
    finally:
        os.chdir(previous)


def segment(ct_path, out_path, info_path, device, tmp):
    t0 = time.perf_counter()
    # Everything TotalSegmentator writes temporarily goes where the job can clean it up.
    os.makedirs(tmp, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = tmp
    tempfile.tempdir = tmp

    out("PROG", 3, "Loading TotalSegmentator")
    import numpy as np
    import SimpleITK as sitk
    import torch
    import totalsegmentator.python_api as api
    from importlib.metadata import version
    from totalsegmentator.map_to_binary import class_map
    # TotalSegmentator's usage statistics are skipped for runs started by this program.
    # Its own settings file is left unchanged.
    if hasattr(api, "send_usage_stats"):
        api.send_usage_stats = lambda *a, **k: None

    use_gpu = device == "gpu"
    if device == "auto" and torch.cuda.is_available():
        use_gpu = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 >= MIN_GPU_GB
    use_gpu = use_gpu and torch.cuda.is_available()
    run_device = "gpu" if use_gpu else "cpu"

    out("PROG", 8, "Reading the CT")
    ct = _read(ct_path)
    ras = sitk.DICOMOrient(ct, "RAS")
    with tempfile.TemporaryDirectory(prefix="liverseg_") as work:
        inp, ml = Path(work) / "input.nii", Path(work) / "ml.nii"
        _write(ras, inp, False)
        out("PROG", 15, f"TotalSegmentator is running ({run_device.upper()})")
        with contextlib.redirect_stdout(sys.stderr):
            api.totalsegmentator(input=str(inp), output=str(ml), task=TASK, roi_subset=[ORGAN],
                                 ml=True, fast=False, device=run_device, quiet=False)
        out("PROG", 85, "Moving the mask onto the CT grid")
        ml_img = _read(ml)
        label = {v: k for k, v in class_map[TASK].items()}[ORGAN]
        arr = (sitk.GetArrayFromImage(ml_img) == label).astype(np.uint8)
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(ml_img)
        mask = sitk.Resample(mask, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    marr = sitk.GetArrayFromImage(mask)
    if not marr.any():
        out("FAIL", "TotalSegmentator found no liver (empty mask)")
        return 3
    tmp_out = out_path.with_name(out_path.name.replace(".nii.gz", ".tmp.nii.gz"))
    _write(mask, tmp_out)
    tmp_out.replace(out_path)
    secs = time.perf_counter() - t0
    sp = ct.GetSpacing()
    info_path.write_text(json.dumps({
        "totalsegmentator_version": version("TotalSegmentator"),
        "task": TASK, "roi_subset": [ORGAN], "fast": False,
        "device": run_device,
        "gpu": torch.cuda.get_device_name(0) if use_gpu else None,
        "voxels": int(marr.sum()),
        "volume_ml": round(float(marr.sum()) * sp[0] * sp[1] * sp[2] / 1000.0, 2),
        "seconds": round(secs, 1),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    out("DONE", f"{secs:.1f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--ct")
    ap.add_argument("--out")
    ap.add_argument("--info")
    ap.add_argument("--tmp")
    ap.add_argument("--device", default="auto", choices=("auto", "gpu", "cpu"))
    a = ap.parse_args()
    if a.probe:
        return probe()
    try:
        return segment(Path(a.ct), Path(a.out), Path(a.info), a.device, a.tmp)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        out("FAIL", f"{type(exc).__name__}: {exc}".replace("|", "/").replace("\n", " ")[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
