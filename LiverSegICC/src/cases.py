"""Case records: each case lives in workspace/cases/<CODE>/.

  ct.nii.gz        LPS-oriented CT (int16)
  meta.json        import details (no patient identifiers; those are only in case_source_map.csv)
  TS.nii.gz        TotalSegmentator liver mask (same grid as the CT)
  ts_info.json     TotalSegmentator version, device, duration
  AUTH1.nii.gz     reader 1's corrected mask
  AUTH1_log.json   status, active time, strokes, voxels added/removed
  AUTH2...         the same for reader 2

A reader sees only TS and their OWN mask (blinding).
"""

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import config as C


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_cases():
    if not C.CASES_DIR.exists():
        return []
    return sorted(p.name for p in C.CASES_DIR.iterdir() if (p / C.META_FILE).exists() and (p / C.CT_FILE).exists())


def case_dir(case_id):
    return C.CASES_DIR / case_id


def next_case_id():
    used = set()
    if C.CASES_DIR.exists():
        used = {p.name for p in C.CASES_DIR.iterdir()}
    i = 1
    while f"C{i:03d}" in used:
        i += 1
    return f"C{i:03d}"


def cases_by_series():
    """{SeriesInstanceUID: case code} for every imported case."""
    out = {}
    for cid in list_cases():
        m = read_json(case_dir(cid) / C.META_FILE, {})
        if m.get("series_uid"):
            out[m["series_uid"]] = cid
    return out


def create_case(series, source_folder):
    """Loads a series and creates a new case. Returns its code."""
    import sitk_io
    from dicom_io import load_series

    C.ensure_dirs()
    img, warnings = load_series(series["files"])
    cid = next_case_id()
    d = case_dir(cid)
    d.mkdir(parents=True)
    try:
        sitk_io.write_image(img, d / C.CT_FILE)
        meta = {
            "case_id": cid,
            "series_uid": series["series_uid"],
            "series_desc": series["desc"],
            "contrast_hint": series["contrast"],
            "slice_thickness_mm": series["thickness"],
            "kvp": series["kvp"],
            "kernel": series["kernel"],
            "manufacturer": series["manufacturer"],
            "model": series["model"],
            "size_xyz": list(img.GetSize()),
            "spacing_xyz": [round(v, 5) for v in img.GetSpacing()],
            "n_files": series["n"],
            "warnings": warnings,
            "imported_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(d / C.META_FILE, meta)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    new = not C.SOURCE_MAP.exists()
    with open(C.SOURCE_MAP, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        if new:
            w.writerow(["case", "source_folder", "patient_id", "study_date", "series_uid",
                        "series_description", "imported_at"])
        w.writerow([cid, source_folder, series["patient_id"], series["study_date"], series["series_uid"],
                    series["desc"], meta["imported_at"]])
    return cid


def delete_case(case_id):
    d = case_dir(case_id)
    if d.exists() and d.parent == C.CASES_DIR:
        shutil.rmtree(d)


def ts_done(case_id):
    return (case_dir(case_id) / C.mask_file("TS")).exists()


def reader_status(case_id, reader):
    """{'status': 'pending'|'in_progress'|'done', 'active_seconds': ...}"""
    log = read_json(case_dir(case_id) / C.reader_log_file(reader), None)
    if log is None:
        return {"status": C.PENDING, "active_seconds": 0.0}
    return log


def reader_order(reader, case_ids):
    """Stable pseudo-random order per reader (reduces order effects; adding a case does not
    change the relative order of the others)."""
    return sorted(case_ids, key=lambda c: hashlib.sha256(f"{reader}|{c}".encode()).hexdigest())
