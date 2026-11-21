"""Shared paths and settings.

Everything the program creates lives in two places inside its own folder:
  workspace/            cases, masks, results and logs (contains patient images: never publish it)
  local_settings.json   per-computer choices (TotalSegmentator location, case folder, last reader)
Deleting the program folder removes all of it.
"""

import json
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
WORK_DIR = Path(os.environ.get("LIVERSEG_WORKDIR", str(ROOT_DIR / "workspace")))
CASES_DIR = WORK_DIR / "cases"
RESULTS_DIR = WORK_DIR / "results"
LOG_DIR = WORK_DIR / "logs"
SOURCE_MAP = WORK_DIR / "case_source_map.csv"   # case code <-> patient/folder (administrator only)
SCAN_CACHE = WORK_DIR / "folder_scan.json"      # what was found in the case folder
LOCAL_SETTINGS = ROOT_DIR / "local_settings.json"

# Background segmentation job
JOB_LOCK = WORK_DIR / "job.lock"          # one segmentation job at a time
JOB_STATUS = WORK_DIR / "job_status.json"
JOB_QUEUE = WORK_DIR / "job_queue.json"
JOB_STOP = WORK_DIR / "job.stop"          # "after" = stop after this case, "now" = abort this case

READERS = ("AUTH1", "AUTH2")
MASKS = ("TS", "AUTH1", "AUTH2")
PAIRS = (("TS", "AUTH1"), ("TS", "AUTH2"), ("AUTH1", "AUTH2"))

# Files inside a case folder
CT_FILE = "ct.nii.gz"
META_FILE = "meta.json"
TS_INFO_FILE = "ts_info.json"

# Reader progress values stored in <reader>_log.json
PENDING, IN_PROGRESS, DONE = "pending", "in_progress", "done"

TS_TASK = "total"
TS_ORGAN = "liver"
REFERENCE_TS_VERSION = "2.13.0"           # version installed by the "Install" button
TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TS_HOME_URL = "https://github.com/wasserth/TotalSegmentator#installation"


def mask_file(name):
    return f"{name}.nii.gz"


def reader_log_file(reader):
    return f"{reader}_log.json"


def ensure_dirs():
    for d in (WORK_DIR, CASES_DIR, RESULTS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings():
    try:
        return json.loads(LOCAL_SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(settings):
    tmp = LOCAL_SETTINGS.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LOCAL_SETTINGS)
