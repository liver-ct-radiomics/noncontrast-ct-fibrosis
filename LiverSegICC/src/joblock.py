"""Single-job lock, so two segmentation jobs never compete for the GPU or case codes.
The lock file carries its owner's PID; if that process is gone the lock is stale and is
taken over.
"""

import json
import os
import sys
from datetime import datetime

import config as C


def pid_alive(pid):
    if not pid or pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    # On Windows os.kill(pid, 0) KILLS the process, so the state is only queried.
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def holder():
    """{'pid','owner','since'} if a live process holds the lock, otherwise None."""
    try:
        info = json.loads(C.JOB_LOCK.read_text(encoding="utf-8"))
    except Exception:
        return None
    return info if pid_alive(int(info.get("pid", 0))) else None


def acquire(owner):
    C.ensure_dirs()
    for _ in range(2):
        try:
            fd = os.open(str(C.JOB_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if holder() is not None:
                return False
            try:
                C.JOB_LOCK.unlink()  # stale lock
            except OSError:
                return False
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "owner": owner,
                       "since": datetime.now().isoformat(timespec="seconds")}, fh)
        return True
    return False


def release():
    try:
        info = json.loads(C.JOB_LOCK.read_text(encoding="utf-8"))
        if int(info.get("pid", 0)) == os.getpid():
            C.JOB_LOCK.unlink()
    except Exception:
        pass
