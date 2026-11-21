"""Finds the cases in a folder: one subfolder per case, DICOM files anywhere inside it.

Results are cached in workspace/folder_scan.json, so the list appears at once the next time
the program opens and only new or changed folders are read again. The cache holds series
descriptions and folder names, never patient identifiers read from DICOM headers.
"""

from pathlib import Path

import config as C
import dicom_io
from cases import read_json, write_json

_SERIES_FIELDS = ("series_uid", "desc", "n", "thickness", "contrast", "kernel", "usable")


def case_folders(root):
    """Subfolders of root. A lone wrapper folder (as left by unzipping) is stepped into."""
    current = Path(root)
    while True:
        entries = list(current.iterdir())
        subs = sorted((p for p in entries if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: p.name.lower())
        files = [p for p in entries if p.is_file()]
        if len(subs) == 1 and not files:
            current = subs[0]
            continue
        return subs if subs else [current]


def scan_case_folder(folder):
    series = dicom_io.scan_folder(folder)
    best, ambiguous = dicom_io.auto_pick(series)
    return {
        "path": str(folder),
        "name": folder.name,
        "mtime": folder.stat().st_mtime,
        "series": [{k: s[k] for k in _SERIES_FIELDS} for s in series],
        "series_uid": best["series_uid"] if best else None,
        # A series that looks contrast-enhanced is also shown to the user before it is used.
        "needs_review": bool(series) and (ambiguous or best is None or best["contrast"] == "yes"),
        "n_patients": len({s["patient_id"] for s in series}),
        "n_studies": len({s["study_uid"] for s in series}),
    }


def load_cache(root):
    cache = read_json(C.SCAN_CACHE, None) or {}
    if cache.get("root") != str(root):
        return {"root": str(root), "rows": {}}
    return cache


def load_rows(root):
    rows = load_cache(root)["rows"]
    return [rows[k] for k in sorted(rows, key=lambda k: rows[k]["name"].lower())]


def scan_root(root, force = False, progress=None, row_done=None):
    """Reads new or changed case folders. Returns how many were read."""
    C.ensure_dirs()
    root = Path(root)
    cache = load_cache(root)
    rows = cache["rows"]
    folders = case_folders(root)
    current = {str(f) for f in folders}
    for gone in [k for k in rows if k not in current]:
        del rows[gone]
    todo = [f for f in folders
            if force or str(f) not in rows or rows[str(f)].get("mtime") != f.stat().st_mtime]
    for i, folder in enumerate(todo, 1):
        if progress:
            progress(i, len(todo))
        try:
            row = scan_case_folder(folder)
        except Exception as exc:
            row = {"path": str(folder), "name": folder.name, "mtime": 0, "series": [], "series_uid": None,
                   "needs_review": False, "n_patients": 0, "n_studies": 0, "error": type(exc).__name__}
        old = rows.get(str(folder))
        if old and old.get("confirmed") and any(s["series_uid"] == old["series_uid"] for s in row["series"]):
            # Keep the series the user picked by hand
            row.update(series_uid=old["series_uid"], needs_review=False, confirmed=True)
        rows[str(folder)] = row
        write_json(C.SCAN_CACHE, cache)
        if row_done:
            row_done()
    write_json(C.SCAN_CACHE, cache)
    return len(todo)


def set_series(root, path, series_uid):
    cache = load_cache(root)
    row = cache["rows"].get(path)
    if row:
        row.update(series_uid=series_uid, needs_review=False, confirmed=True)
        write_json(C.SCAN_CACHE, cache)
