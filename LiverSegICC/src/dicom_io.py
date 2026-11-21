"""DICOM scanning, series selection and conversion to a volume.

Slices are sorted along the slice normal from feet to head, and a broken z spacing is
recovered from the slice positions. The volume is then reoriented losslessly to LPS
(DICOMOrient), so array axes mean the same in every case: x -> patient left, y -> posterior,
z -> head. An axial slice of this array is directly in radiological orientation (patient
right on the left of the screen, anterior at the top).
"""

import os
import re
from pathlib import Path

import numpy as np

SCAN_TAGS = [
    "SOPClassUID", "Modality", "SeriesInstanceUID", "StudyInstanceUID", "SeriesDescription",
    "SeriesNumber", "SliceThickness", "ImageOrientationPatient", "ImagePositionPatient",
    "ImageType", "ContrastBolusAgent", "KVP", "ConvolutionKernel", "Manufacturer",
    "ManufacturerModelName", "StudyDate", "PatientID", "Rows", "Columns", "PixelSpacing",
    "ProtocolName", "BodyPartExamined", "StudyDescription",
]
SKIP_SUFFIXES = (".json", ".txt", ".csv", ".xml", ".jpg", ".png", ".pdf", ".ini", ".db")

# Series descriptions in the wild mix English and Turkish terms.
_NONCONTRAST = re.compile(
    r"non[\s_\-]*kontrast|non[\s_\-]*contrast|kontrasts[iı]z|nonkon|nativ|unenhanced|"
    r"pre[\s_\-]*kontrast|pre[\s_\-]*contrast|plain|\bnc\b|\bnon\b|-\s*c\b|kontrastsiz",
    re.IGNORECASE,
)
_CONTRAST = re.compile(
    r"\+\s*c\b|\bc\s*\+|kontrastl|post[\s_\-]*kontrast|post[\s_\-]*contrast|with[\s_\-]*contrast|"
    r"arter|arterial|portal|ven[oö]z|venous|ge[cç]\s*faz|delayed|nefrogra|\d+\s*\.?\s*sn\b|\d+\s*sec\b|\bsn\b",
    re.IGNORECASE,
)


def contrast_hint(desc, agent):
    """'no' (description says non-contrast), 'yes' (signs of contrast), '?' (unclear)."""
    d = desc or ""
    if _NONCONTRAST.search(d):
        return "no"
    if (agent or "").strip() or _CONTRAST.search(d):
        return "yes"
    return "?"


def _is_axial(iop):
    try:
        r = np.array([float(v) for v in iop[:3]])
        c = np.array([float(v) for v in iop[3:6]])
        n = np.cross(r, c)
        return bool(abs(n[2]) > 0.9)
    except Exception:
        return False


def _normal(iop):
    r = np.array([float(v) for v in iop[:3]])
    c = np.array([float(v) for v in iop[3:6]])
    return np.cross(r, c)


def scan_folder(root, progress=None):
    """Scans a folder and its subfolders, grouping files by SeriesInstanceUID."""
    import pydicom

    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    series = {}
    for i, f in enumerate(files):
        if progress and i % 200 == 0:
            progress(i, len(files))
        if f.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=SCAN_TAGS)
        except Exception:
            continue
        uid = str(ds.get("SeriesInstanceUID", "") or "")
        if not uid:
            continue
        s = series.get(uid)
        if s is None:
            itype = ds.get("ImageType", None)
            s = {
                "series_uid": uid,
                "study_uid": str(ds.get("StudyInstanceUID", "") or ""),
                "modality": str(ds.get("Modality", "") or ""),
                "desc": str(ds.get("SeriesDescription", "") or ""),
                "number": str(ds.get("SeriesNumber", "") or ""),
                "thickness": _f(ds.get("SliceThickness", None)),
                "iop": [float(v) for v in ds.get("ImageOrientationPatient", [])] or None,
                "image_type": "\\".join(str(x) for x in itype) if itype else "",
                "agent": str(ds.get("ContrastBolusAgent", "") or ""),
                "kvp": _f(ds.get("KVP", None)),
                "kernel": _join(ds.get("ConvolutionKernel", "")),
                "manufacturer": str(ds.get("Manufacturer", "") or ""),
                "model": str(ds.get("ManufacturerModelName", "") or ""),
                "study_date": str(ds.get("StudyDate", "") or ""),
                "patient_id": str(ds.get("PatientID", "") or ""),
                "pixel_spacing": [float(v) for v in ds.get("PixelSpacing", [])] or None,
                "study_desc": str(ds.get("StudyDescription", "") or ""),
                "protocol": str(ds.get("ProtocolName", "") or ""),
                "files": [],
            }
            series[uid] = s
        s["files"].append(str(f))
    out = list(series.values())
    for s in out:
        s["n"] = len(s["files"])
        s["axial"] = bool(s["iop"]) and _is_axial(s["iop"])
        s["contrast"] = contrast_hint(s["desc"], s["agent"])
        it = s["image_type"].upper()
        s["usable"] = (
            s["modality"] == "CT" and s["axial"] and s["n"] >= 40
            and "LOCALIZER" not in it and "SCOUT" not in s["desc"].upper()
        )
    out.sort(key=lambda s: (not s["usable"], _rank(s)))
    return out


def _rank(s):
    # non-contrast > unclear > contrast; thin slices; many slices
    c = {"no": 0, "?": 1, "yes": 2}[s["contrast"]]
    th = s["thickness"] if s["thickness"] else 99.0
    return (c, th, -s["n"])


def auto_pick(series):
    """(chosen series, ambiguous). Ambiguous: more than one usable candidate and the best one's
    description does not state that it is non-contrast."""
    usable = [s for s in series if s["usable"]]
    if not usable:
        return None, True
    best = min(usable, key=_rank)
    noncon = [s for s in usable if s["contrast"] == "no"]
    ambiguous = len(usable) > 1 and not (best["contrast"] == "no" and len(noncon) == 1)
    return best, ambiguous


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def _join(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return "\\".join(str(x) for x in v)
    return str(v)


def load_series(files):
    """File list -> (LPS-oriented SimpleITK int16 image, list of warnings)."""
    import pydicom
    import SimpleITK as sitk

    entries = []
    normal = None
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"])
            if normal is None and "ImageOrientationPatient" in ds:
                normal = _normal(ds.ImageOrientationPatient)
            ipp = [float(v) for v in ds.ImagePositionPatient]
            entries.append((ipp, fp))
        except Exception:
            continue
    if len(entries) < 3:
        raise RuntimeError("The series does not have enough slices")
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    # Position along the slice normal; if two slices share a position, drop one
    pos = [(float(np.dot(normal, ipp)), fp) for ipp, fp in entries]
    pos.sort(key=lambda t: t[0])
    uniq = []
    for p, fp in pos:
        if uniq and abs(p - uniq[-1][0]) < 1e-3:
            continue
        uniq.append((p, fp))
    warnings = []
    if len(uniq) != len(pos):
        warnings.append(f"{len(pos) - len(uniq)} duplicate slices dropped")
    z = np.array([p for p, _ in uniq])
    d = np.diff(z)
    if d.size:
        med = float(np.median(d))
        if med > 0 and (np.abs(d - med) > 0.1 * med).any():
            warnings.append(f"uneven slice spacing (median {med:.3f} mm, min {d.min():.3f}, max {d.max():.3f})")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([fp for _, fp in uniq])
    img = reader.Execute()
    sx, sy, sz = img.GetSpacing()
    if d.size and (sz <= 1e-3 or abs(sz - float(np.median(d))) > 0.01):
        sz_new = round(float(np.median(d)), 4)
        img.SetSpacing((sx, sy, sz_new))
        warnings.append(f"z spacing corrected from {sz:.4f} to {sz_new} mm")
    if img.GetPixelID() not in (sitk.sitkInt16,):
        arr = sitk.GetArrayFromImage(img)
        arr = np.clip(np.rint(arr), -32768, 32767).astype(np.int16)
        img2 = sitk.GetImageFromArray(arr)
        img2.CopyInformation(img)
        img = img2
    img = sitk.DICOMOrient(img, "LPS")
    return img, warnings
