"""Background segmentation job: imports the queued cases and runs TotalSegmentator on each.

The main window starts this as a separate process, so it continues if the window is closed.
Each case finished appears in the readers' lists at once.

  Queue  : workspace/job_queue.json   (the window may append while the job runs)
  Status : workspace/job_status.json  (the window reads it every 2 s)
  Stop   : workspace/job.stop         ("after" = stop after this case, "now" = abort this case)
  Lock   : workspace/job.lock

Logs contain case codes only, never folder names or patient identifiers.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import config as C
import cases
import dicom_io
import joblock
import ts_locate

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
BELOW_NORMAL_PRIORITY = 0x00004000 if sys.platform == "win32" else 0  # keeps the computer usable
KILLED = -999

STATUS = {}


def say(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def status(**kw):
    STATUS.update(kw, updated=datetime.now().isoformat(timespec="seconds"))
    # The window may be reading the file at this moment: retry briefly on Windows
    for _ in range(10):
        try:
            cases.write_json(C.JOB_STATUS, STATUS)
            return
        except OSError:
            time.sleep(0.05)


def mark(key, state, **kw):
    STATUS["items"][key] = {"state": state, **kw}
    status()


def stop_request():
    try:
        return C.JOB_STOP.read_text(encoding="utf-8").strip() or "after"
    except OSError:
        return ""


def clean_tmp(tmp):
    """Removes temporary images, including those left behind by an aborted case."""
    for child in tmp.glob("*"):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def import_case(item):
    found = [s for s in dicom_io.scan_folder(item["folder"]) if s["series_uid"] == item["series_uid"]]
    if not found:
        raise RuntimeError("the chosen series was not found in the folder")
    return cases.create_case(found[0], item["folder"])


def run_ts(cid, python, device, tmp, errlog):
    d = cases.case_dir(cid)
    args = [python, str(C.SRC_DIR / "ts_worker.py"), "--ct", str(d / C.CT_FILE),
            "--out", str(d / C.mask_file("TS")), "--info", str(d / C.TS_INFO_FILE),
            "--tmp", str(tmp), "--device", device]
    p = subprocess.Popen(args, cwd=str(C.WORK_DIR), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errlog,
                         env=ts_locate.python_env(python), text=True, encoding="utf-8", errors="replace",
                         bufsize=1, creationflags=CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY)
    last_fail = {"msg": ""}

    def reader():
        for line in p.stdout:
            parts = line.rstrip("\n").split("|")
            if parts[0] == "PROG" and len(parts) >= 3:
                status(case_percent=int(parts[1]), message=parts[2])
            elif parts[0] == "FAIL":
                last_fail["msg"] = "|".join(parts[1:])

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    while p.poll() is None:
        if stop_request() == "now":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True,
                           creationflags=CREATE_NO_WINDOW)
            p.wait()
            t.join(timeout=5)
            return KILLED, "aborted"
        time.sleep(0.5)
    t.join(timeout=5)
    return p.returncode, last_fail["msg"]


def process(item, k, python, device, tmp, errlog):
    """Handles one queue item. Returns False if the user aborted it."""
    key = item["key"]
    status(i=k, current_key=key, current_case="", case_started=datetime.now().isoformat(timespec="seconds"),
           case_percent=0, message="Importing the DICOM series")
    mark(key, "importing")
    cid = item.get("case") or cases.cases_by_series().get(item["series_uid"])
    if cid is None:
        try:
            cid = import_case(item)
        except Exception as exc:
            # Source folder names may contain patient identifiers: keep them out of the log
            errlog.write(traceback.format_exc().replace(item["folder"], "<case folder>"))
            mark(key, "failed", error=f"import: {type(exc).__name__}")
            STATUS["failed"] += 1
            say(f"#{k} import FAILED ({type(exc).__name__})")
            return True
        say(f"#{k} -> {cid} imported")
    status(current_case=cid)
    if cases.ts_done(cid):
        mark(key, "done", case=cid)
        return True
    mark(key, "running", case=cid)
    attempts = [device] if device == "cpu" else [device, "cpu"]  # out of GPU memory: try the CPU
    rc, msg = 1, ""
    for attempt in attempts:
        rc, msg = run_ts(cid, python, attempt, tmp, errlog)
        clean_tmp(tmp)
        if rc == 0:
            break
        if rc == KILLED:
            mark(key, "stopped", case=cid)
            say(f"{cid} aborted by the user")
            return False
        if rc == 3:
            break
        say(f"{cid} failed on '{attempt}': {msg}")
    if rc == 0:
        secs = (cases.read_json(cases.case_dir(cid) / C.TS_INFO_FILE, {}) or {}).get("seconds", 0)
        mark(key, "done", case=cid)
        STATUS["done"] += 1
        say(f"{cid} segmented ({secs:.0f} s)")
    else:
        mark(key, "failed", case=cid, error=msg or "see the log")
        STATUS["failed"] += 1
        say(f"{cid} FAILED: {msg}")
    return True


def main():
    C.ensure_dirs()
    if not joblock.acquire("segmentation job"):
        say("Another job is running; exiting.")
        return 2
    try:
        C.JOB_STOP.unlink()
    except FileNotFoundError:
        pass
    status(running=True, pid=os.getpid(), started=datetime.now().isoformat(timespec="seconds"), phase="running",
           i=0, n=0, current_key="", current_case="", case_started="", case_percent=0, message="Starting",
           items={}, done=0, failed=0)
    tmp = ts_locate.ascii_tmp_dir()
    rc = 0
    errlog_path = C.LOG_DIR / f"job_{datetime.now():%Y%m%d_%H%M%S}_details.log"
    try:
        with open(errlog_path, "w", encoding="utf-8", errors="replace", buffering=1) as errlog:
            handled = set()
            k = 0
            while True:
                queue = cases.read_json(C.JOB_QUEUE, None) or {"items": []}
                items = queue.get("items", [])
                status(n=len(items))
                pending = [it for it in items if it["key"] not in handled]
                if not pending:
                    break
                if stop_request():
                    status(phase="stopped", message="Stopped by the user")
                    say("Stop requested: exiting.")
                    return 0
                item = pending[0]
                handled.add(item["key"])
                k += 1
                if not process(item, k, queue["python"], queue.get("device", "auto"), tmp, errlog):
                    status(phase="stopped", message="Stopped by the user")
                    return 0
        status(phase="finished", current_key="", current_case="",
               message=f"{STATUS['done']} segmented, {STATUS['failed']} failed")
        say("Finished.")
        return rc
    except Exception as exc:
        with open(errlog_path, "a", encoding="utf-8", errors="replace") as fh:
            traceback.print_exc(file=fh)
        status(phase="error", message=f"{type(exc).__name__}: see the log")
        say(f"ERROR: {type(exc).__name__}")
        return 1
    finally:
        clean_tmp(tmp)
        status(running=False)
        joblock.release()
        for f in (C.JOB_STOP, C.JOB_QUEUE):
            try:
                Path(f).unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
