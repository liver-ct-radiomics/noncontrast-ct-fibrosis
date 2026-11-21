"""Finds, checks and installs TotalSegmentator.

TotalSegmentator may already be on the computer in some other Python installation, so the
program looks for it there first and borrows that installation. Nothing is written into a
borrowed installation (bytecode caching is switched off for it).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config as C

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
MAX_DEEP_CHECKS = 4
# PyTorch's longest file adds about 130 characters to the program folder's path, and Windows
# limits a path to 260 unless long paths have been enabled.
MAX_ROOT_PATH = 125
_QUICK = ("import importlib.util as u, importlib.metadata as m, sys;"
          "s = u.find_spec('totalsegmentator');"
          "print('TS|' + (m.version('TotalSegmentator') if s else '-'))")
_SKIP_DIRS = {"appdata", "node_modules", "site-packages", "__pycache__", "windows", "program files",
              "program files (x86)", "$recycle.bin", "dicom", "workspace"}


def console_python():
    """python.exe next to the running interpreter (pythonw.exe has no usable stdout)."""
    exe = Path(sys.executable)
    twin = exe.with_name("python.exe")
    return str(twin) if exe.name.lower() == "pythonw.exe" and twin.exists() else str(exe)


def own_python():
    return C.ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def python_env(python):
    """Environment for running another Python installation."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    # This program's own virtual environment must not leak into the other interpreter.
    for name in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(name, None)
    prefix = Path(python).parent
    extra = [str(prefix), str(prefix / "Scripts"), str(prefix / "Library" / "bin")]  # conda needs these
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _run(args, timeout=15, env=None):
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env, creationflags=CREATE_NO_WINDOW,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout or ""


def quick_check(python):
    """TotalSegmentator version in this Python, or None. Takes a fraction of a second."""
    for line in _run([str(python), "-c", _QUICK], env=python_env(python)).splitlines():
        if line.startswith("TS|") and line[3:] != "-":
            return line[3:].strip()
    return None


def probe(python, timeout=240):
    """Imports TotalSegmentator and torch in that Python: the real test of an installation."""
    text = _run([str(python), str(C.SRC_DIR / "ts_worker.py"), "--probe"], timeout=timeout, env=python_env(python))
    for line in text.splitlines():
        if line.startswith("INFO|"):
            try:
                return {"python": str(python), **json.loads(line[5:])}
            except ValueError:
                break
    return {"python": str(python), "ok": False, "error": "the check did not finish"}


def _python_for_script(script):
    """python.exe that belongs to a Scripts\\TotalSegmentator.exe."""
    for p in (script.with_name("python.exe"), script.parent.parent / "python.exe"):
        if p.is_file():
            return p
    return None


def python_from_user_choice(path):
    p = Path(path)
    if p.name.lower() in ("python.exe", "pythonw.exe", "pythonslicer.exe"):
        return p.with_name("python.exe") if p.name.lower() == "pythonw.exe" else p
    return _python_for_script(p)


def _find_virtualenvs(budget_s = 8.0):
    """Breadth-first search of the user's folders for virtual environments, with a time limit."""
    deadline = time.monotonic() + budget_s
    queue = deque([(Path.home(), 0)])
    while queue and time.monotonic() < deadline:
        folder, depth = queue.popleft()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        if any(e.name == "pyvenv.cfg" for e in entries):
            yield Path(folder) / "Scripts" / "python.exe"
            continue
        if depth >= 5:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False) and not e.name.startswith(".") \
                        and e.name.lower() not in _SKIP_DIRS:
                    queue.append((e.path, depth + 1))
            except OSError:
                pass


def candidate_pythons():
    out = []
    seen = set()

    def add(p):
        if not p:
            return
        p = Path(p)
        if p.name.lower() == "pythonw.exe":
            p = p.with_name("python.exe")
        key = os.path.normcase(str(p))
        if key in seen or "windowsapps" in key or not p.is_file():
            return
        seen.add(key)
        out.append(p)

    add(own_python())
    for line in _run(["where", "TotalSegmentator"]).splitlines():
        add(_python_for_script(Path(line.strip())))
    for line in _run(["py", "-0p"]).splitlines():
        m = re.search(r"[A-Za-z]:\\.*\.exe", line)
        if m:
            add(m.group(0))
    for line in _run(["where", "python"]).splitlines():
        add(line.strip())
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    for pattern_root, pattern in ((local / "Programs" / "Python", "Python3*"), (Path("C:/"), "Python3*"),
                                  (Path(os.environ.get("ProgramFiles", "C:/Program Files")), "Python3*")):
        for d in sorted(pattern_root.glob(pattern)):
            add(d / "python.exe")
    # conda: its own list of environments, then the usual install folders
    try:
        for line in (home / ".conda" / "environments.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                add(Path(line.strip()) / "python.exe")
    except OSError:
        pass
    for base in (home, Path("C:/"), Path("C:/ProgramData")):
        for name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
            add(base / name / "python.exe")
            for env_dir in sorted((base / name / "envs").glob("*")):
                add(env_dir / "python.exe")
    # 3D Slicer's Python, where its TotalSegmentator extension installs the package
    for vendor in ("slicer.org", "NA-MIC"):
        for d in sorted((local / vendor).glob("Slicer*")):
            add(d / "bin" / "PythonSlicer.exe")
    for p in _find_virtualenvs():
        add(p)
    return out


def _version_key(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "0")[:3])


def search():
    """All installations found, each fully checked (newest first, at most MAX_DEEP_CHECKS)."""
    found = []
    with ThreadPoolExecutor(8) as ex:
        pythons = candidate_pythons()
        for py, ver in zip(pythons, ex.map(quick_check, pythons)):
            if ver:
                found.append((py, ver))
    found.sort(key=lambda t: _version_key(t[1]), reverse=True)
    with ThreadPoolExecutor(3) as ex:
        return list(ex.map(lambda t: {"version": t[1], **probe(t[0])}, found[:MAX_DEEP_CHECKS]))


def usable(info):
    return bool(info.get("ok")) and _version_key(info.get("version"))[:1] >= (2,)


def pick_best(found):
    """Prefers a working GPU, then the reference version, then the newest."""
    ok = [f for f in found if usable(f)]
    if not ok:
        return None
    return max(ok, key=lambda f: (bool(f.get("cuda")), f.get("version") == C.REFERENCE_TS_VERSION,
                                  _version_key(f.get("version"))))


def nvidia_present():
    return bool(shutil.which("nvidia-smi")) and bool(_run(["nvidia-smi", "-L"]).strip())


def install_commands(use_gpu):
    py = str(own_python())
    index = "https://download.pytorch.org/whl/" + ("cu124" if use_gpu else "cpu")
    common = ["--disable-pip-version-check", "--no-input", "--progress-bar", "off"]
    constraints = str(C.ROOT_DIR / "constraints-totalsegmentator.txt")
    return [
        [py, "-m", "pip", "install", *common, f"torch=={C.TORCH_VERSION}",
         f"torchvision=={C.TORCHVISION_VERSION}", "--index-url", index],
        [py, "-m", "pip", "install", *common, f"TotalSegmentator=={C.REFERENCE_TS_VERSION}", "-c", constraints],
    ]


def ascii_tmp_dir():
    """Temporary folder for TotalSegmentator, kept inside the workspace whenever possible.
    Its path must be plain ASCII (see sitk_io.py); on a non-ASCII path the short 8.3 form of the
    same folder is used, and the public folder is the last resort."""
    base = C.WORK_DIR / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    if str(base).isascii():
        return base
    if sys.platform == "win32":
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(base), buf, 1024) and buf.value.isascii():
            return Path(buf.value)
    fallback = Path(os.environ.get("PUBLIC", "C:/Users/Public")) / "LiverSegICC_tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback
