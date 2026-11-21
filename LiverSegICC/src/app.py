"""LiverSegICC main window.

Tabs:
  1. Cases       choose the case folder, tick cases, segment them with TotalSegmentator in the background
  2. Correction  AUTH1 / AUTH2 open their own list and correct the mask with a brush and an eraser
  3. Analysis    volume, mean density, ICC, Dice, IoU for TS vs AUTH1, TS vs AUTH2, AUTH1 vs AUTH2
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import config as C
import cases
import folder_scan
import joblock
import ts_locate
from uiutil import enable_dpi_awareness, px

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
WORKER_PY = ts_locate.console_python()
READER_STATUS_TEXT = {C.PENDING: "pending", C.IN_PROGRESS: "in progress", C.DONE: "DONE"}
DEVICES = {"Auto": "auto", "GPU": "gpu", "CPU": "cpu"}


def fmt_min(sec):
    try:
        return f"{float(sec) / 60:.1f}"
    except (TypeError, ValueError):
        return ""


def fmt_duration(sec):
    sec = int(max(0, sec))
    return f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s"


def worker_env():
    return dict(os.environ, PYTHONIOENCODING="utf-8", LIVERSEG_WORKDIR=str(C.WORK_DIR),
                PYTHONDONTWRITEBYTECODE="1")


class SeriesDialog(tk.Toplevel):
    """Choose one of a case's series."""

    def __init__(self, master, title, series, preselect_uid):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.result = None
        self.series = series
        ttk.Label(self, padding=8, justify=tk.LEFT, text=(
            "Choose the axial, thin-slice series to segment.\n"
            "The suggested series is highlighted. 'Contrast' is a guess from the series description: please check it."
        )).pack(fill=tk.X)
        cols = ("desc", "n", "th", "con", "kernel", "ok")
        tv = ttk.Treeview(self, columns=cols, show="headings", height=min(12, max(4, len(series))), selectmode="browse")
        for c, t, w in (("desc", "Series description", 300), ("n", "Slices", 60), ("th", "Thickness (mm)", 100),
                        ("con", "Contrast", 70), ("kernel", "Kernel", 90), ("ok", "Suitable", 70)):
            tv.heading(c, text=t)
            tv.column(c, width=px(self, w), anchor=tk.W)
        for i, s in enumerate(series):
            tv.insert("", tk.END, iid=str(i), values=(
                s["desc"] or "(unnamed)", s["n"], s["thickness"] or "", s["contrast"], s["kernel"],
                "yes" if s["usable"] else "no"))
            if s["series_uid"] == preselect_uid:
                tv.selection_set(str(i))
                tv.see(str(i))
        tv.pack(fill=tk.BOTH, expand=True, padx=8)
        tv.bind("<Double-1>", lambda e: self._ok())
        self.tv = tv
        bf = ttk.Frame(self, padding=8)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="Use this series", command=self._ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bf, text="Skip this case", command=self.destroy).pack(side=tk.RIGHT)
        self.grab_set()
        self.focus_force()

    def _ok(self):
        sel = self.tv.selection()
        if not sel:
            return
        s = self.series[int(sel[0])]
        if not s["usable"] and not messagebox.askyesno(
                "Unsuitable series", "This series does not look like an axial CT with enough slices. Use it anyway?",
                parent=self):
            return
        if s["contrast"] == "yes" and not messagebox.askyesno(
                "May be contrast-enhanced", "The description suggests a contrast-enhanced phase. Use it anyway?",
                parent=self):
            return
        self.result = s["series_uid"]
        self.destroy()


class TSDialog(tk.Toplevel):
    """Shows the TotalSegmentator installations found and offers ways to install one."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("TotalSegmentator")
        self.transient(app)
        self.geometry(f"{px(self, 760)}x{px(self, 620)}")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.lbl = ttk.Label(self, padding=10, justify=tk.LEFT, wraplength=px(self, 730))
        self.lbl.pack(fill=tk.X)

        found = ttk.LabelFrame(self, text="Found on this computer", padding=6)
        found.pack(fill=tk.X, padx=10)
        cols = ("ver", "gpu", "path")
        self.tv = ttk.Treeview(found, columns=cols, show="headings", height=3, selectmode="browse")
        for c, t, w in (("ver", "Version", 70), ("gpu", "GPU", 230), ("path", "Python", 410)):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=px(self, w), anchor=tk.W)
        self.tv.pack(fill=tk.X)
        bf = ttk.Frame(found)
        bf.pack(fill=tk.X, pady=(6, 0))
        self.btn_use = ttk.Button(bf, text="Use selected", command=self._use)
        self.btn_use.pack(side=tk.LEFT, padx=2)
        self.btn_search = ttk.Button(bf, text="Search again", command=lambda: self.app.ts_search(force=True))
        self.btn_search.pack(side=tk.LEFT, padx=2)
        self.btn_locate = ttk.Button(bf, text="Locate python.exe or TotalSegmentator.exe…", command=self._locate)
        self.btn_locate.pack(side=tk.LEFT, padx=2)

        inst = ttk.LabelFrame(self, text="Install", padding=6)
        inst.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        ttk.Label(inst, justify=tk.LEFT, text="Installation page:").pack(anchor=tk.W)
        link = tk.Label(inst, text=C.TS_HOME_URL, fg="#0b57d0", cursor="hand2",
                        font=tkfont.Font(underline=True))
        link.pack(anchor=tk.W)
        link.bind("<Button-1>", lambda e: webbrowser.open(C.TS_HOME_URL))
        self.lbl_install = ttk.Label(inst, justify=tk.LEFT, wraplength=px(self, 700), padding=(0, 6))
        self.lbl_install.pack(fill=tk.X)
        self.btn_install = ttk.Button(inst, text="Install into this program's folder", command=self._install)
        self.btn_install.pack(anchor=tk.W)
        self.log = ScrolledText(inst, height=9, state=tk.DISABLED, font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        cf = ttk.Frame(self, padding=(10, 0, 10, 10))
        cf.pack(fill=tk.X)
        ttk.Button(cf, text="Close", command=self._close).pack(side=tk.RIGHT)
        self.use_gpu = ts_locate.nvidia_present()
        self.lbl_install.configure(text=(
            "The button below installs TotalSegmentator "
            f"{C.REFERENCE_TS_VERSION} and PyTorch into this program's own folder, so deleting the folder removes "
            "them too. "
            + ("An NVIDIA graphics card was detected: the GPU build is used (about 3 GB to download, 5 GB on disk). "
               if self.use_gpu else
               "No NVIDIA graphics card was detected: the CPU build is used (about 500 MB to download, 1.5 GB on disk). ")
            + "Internet access is needed now, and once more for about 400 MB of model files on the first segmentation."
        ))
        self.update_view()
        self.focus_force()

    def _append(self, line):
        if not self.winfo_exists():
            return
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def update_view(self):
        if not self.winfo_exists():
            return
        app = self.app
        self.tv.delete(*self.tv.get_children())
        for i, f in enumerate(app.ts_found):
            gpu = f"{f.get('gpu')} ({f.get('vram_gb')} GB)" if f.get("cuda") else "none (CPU)"
            if not ts_locate.usable(f):
                gpu = "not usable: " + str(f.get("error") or "version 2 or newer is needed")
            self.tv.insert("", tk.END, iid=str(i), values=(f.get("version", "?"), gpu, f.get("python", "")))
            if app.ts and f.get("python") == app.ts.get("python"):
                self.tv.selection_set(str(i))
        busy = app.ts_state == "searching" or app.installing
        for b in (self.btn_use, self.btn_search, self.btn_locate, self.btn_install):
            b.configure(state=tk.DISABLED if busy else tk.NORMAL)
        if app.installing:
            self.lbl.configure(text="Installing TotalSegmentator. This can take a while; please keep this window open.")
        elif app.ts_state == "searching":
            self.lbl.configure(text="Searching this computer for TotalSegmentator…")
        elif app.ts:
            self.lbl.configure(text=f"TotalSegmentator {app.ts.get('version')} is ready to use.")
        else:
            self.lbl.configure(text=(
                "TotalSegmentator was not found on this computer. It is needed to segment new cases; correcting "
                "cases that are already segmented and running the analysis work without it."))

    def _use(self):
        sel = self.tv.selection()
        if not sel:
            return
        f = self.app.ts_found[int(sel[0])]
        if not ts_locate.usable(f):
            messagebox.showwarning("Not usable", "This installation cannot be used.", parent=self)
            return
        self.app.ts_set(f)
        self.update_view()

    def _locate(self):
        path = filedialog.askopenfilename(parent=self, title="Locate python.exe or TotalSegmentator.exe",
                                          filetypes=[("Programs", "*.exe")])
        if not path:
            return
        py = ts_locate.python_from_user_choice(path)
        if not py or not ts_locate.quick_check(py):
            messagebox.showwarning("Not found", "TotalSegmentator is not installed in that Python.", parent=self)
            return
        self.app.ts_check_one(py)

    def _install(self):
        if not ts_locate.own_python().is_file():
            messagebox.showerror("Setup needed", "Please run setup.bat first.", parent=self)
            return
        if len(str(C.ROOT_DIR)) > ts_locate.MAX_ROOT_PATH:
            messagebox.showerror(
                "Folder path too long",
                "PyTorch contains files with very long names, and Windows limits the length of a path.\n\n"
                f"Move the program to a shorter path, for example C:\\LiverSegICC, run setup.bat there "
                f"and try again.\n\nCurrent folder:\n{C.ROOT_DIR}", parent=self)
            return
        self.app.installing = True
        self.update_view()
        commands = ts_locate.install_commands(self.use_gpu)

        def work():
            ok = True
            for cmd in commands:
                self.app.ui(lambda c=cmd: self._append("> pip install " + " ".join(c[5:])))
                p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", creationflags=CREATE_NO_WINDOW,
                                     env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                self.app.proc = p
                for line in p.stdout:
                    self.app.ui(lambda ln=line: self._append(ln))
                if p.wait() != 0:
                    ok = False
                    break
            self.app.proc = None

            def done():
                self.app.installing = False
                if ok:
                    self._append("Installation finished.")
                    self.app.ts_search(force=True)
                else:
                    self._append("Installation failed. See the messages above.")
                    self.update_view()
            self.app.ui(done)

        threading.Thread(target=work, daemon=True).start()

    def _close(self):
        if self.app.installing:
            messagebox.showinfo("Installing", "Please wait until the installation has finished.", parent=self)
            return
        self.app.ts_dialog = None
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        C.ensure_dirs()
        # On a scaled display, Treeview rows must follow the font height or they overlap
        line = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        ttk.Style(self).configure("Treeview", rowheight=int(line * 1.45))
        self.title("LiverSegICC: liver segmentation reproducibility (TS / AUTH1 / AUTH2)")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{int(sw * 0.72)}x{int(sh * 0.78)}+{int(sw * 0.08)}+{int(sh * 0.06)}")
        self.q = queue.Queue()
        self.settings = C.load_settings()
        self.proc = None
        self.editors = {}
        self.last_result = None
        # TotalSegmentator
        self.ts_state = "searching"
        self.ts = None
        self.ts_found = []
        self.ts_dialog = None
        self._ts_prompted = False
        self.installing = False
        # Cases table
        self.rows = []
        self.checked = set()
        self._iid_key = {}
        self.scanning = False
        # Background job
        self.job = {}
        self._job_key = None
        self._job_pid = 0
        self._failures_logged = set()

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.nb = nb
        self._build_cases_tab(nb)
        self._build_reader_tab(nb)
        self._build_analysis_tab(nb)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._pump)
        self._poll_job(schedule=False)
        self.refresh()
        self._choose_start_tab()
        self.after(200, self._warn_cloud_folder)
        self.after(300, self.ts_search)
        self.after(400, self._startup_scan)
        self.after(2000, self._poll_job)

    # shared
    def ui(self, fn):
        """Runs fn on the window's thread; safe to call from worker threads."""
        self.q.put(fn)

    def _pump(self):
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def log(self, msg, target = "cases"):
        w = self.log_cases if target == "cases" else self.log_an
        w.configure(state=tk.NORMAL)
        w.insert(tk.END, msg.rstrip() + "\n")
        w.see(tk.END)
        w.configure(state=tk.DISABLED)

    def refresh(self):
        self._refresh_cases()
        self._refresh_reader()
        self._refresh_analysis()

    def _save_settings(self):
        try:
            C.save_settings(self.settings)
        except OSError as exc:
            self.log(f"Could not save settings: {exc}")

    def _open_path(self, p):
        try:
            os.startfile(str(p))
        except Exception as exc:
            messagebox.showerror("Could not open", str(exc), parent=self)

    def _choose_start_tab(self):
        """Opens on the reader's list when there is something waiting for that reader."""
        r = self.settings.get("reader")
        if r in C.READERS and any(cases.ts_done(c) and cases.reader_status(c, r).get("status") != C.DONE
                                  for c in cases.list_cases()):
            self.nb.select(1)

    def _warn_cloud_folder(self):
        path = str(C.WORK_DIR).lower()
        roots = [os.environ.get(k, "") for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
        if "onedrive" in path or any(r and path.startswith(r.lower()) for r in roots):
            messagebox.showwarning(
                "Cloud-synced folder",
                f"The workspace is inside a OneDrive folder:\n{C.WORK_DIR}\n\n"
                "CT images saved here may be uploaded to the cloud. Please move the program folder "
                "somewhere that is not synced (for example C:\\LiverSegICC) before adding cases.", parent=self)

    # TotalSegmentator
    def ts_search(self, force = False):
        if self.ts_state == "searching" and force is False and self.ts_found:
            return
        self.ts_state = "searching"
        self._ts_show()
        saved = self.settings.get("ts_python")

        def work():
            found, chosen = [], None
            if saved and not force and ts_locate.quick_check(saved):
                info = ts_locate.probe(saved)
                if ts_locate.usable(info):
                    found, chosen = [info], info
            if chosen is None:
                found = ts_locate.search()
                chosen = ts_locate.pick_best(found)
            self.ui(lambda: self._ts_result(found, chosen))

        threading.Thread(target=work, daemon=True).start()

    def ts_check_one(self, python):
        """A Python chosen by hand."""
        self.ts_state = "searching"
        self._ts_show()

        def work():
            info = ts_locate.probe(python)
            found = [f for f in self.ts_found if f.get("python") != str(python)] + [info]
            self.ui(lambda: self._ts_result(found, info if ts_locate.usable(info) else self.ts))

        threading.Thread(target=work, daemon=True).start()

    def _ts_result(self, found, chosen):
        self.ts_found = found
        self.ts_state = "ready"
        self.ts = None
        if chosen:
            self.ts_set(chosen)
        else:
            self._ts_show()
        if self.ts_dialog is not None:
            self.ts_dialog.update_view()
        elif not chosen and not self._ts_prompted:
            # Nothing to segment on this computer? Then there is no need to interrupt.
            all_cases = cases.list_cases()
            if not all_cases or not all(cases.ts_done(c) for c in all_cases):
                self._ts_prompted = True
                self.open_ts_dialog()

    def ts_set(self, info):
        self.ts = info
        self.settings["ts_python"] = info.get("python")
        self._save_settings()
        self._ts_show()

    def _ts_show(self):
        if self.ts_state == "searching":
            self.lbl_ts.configure(text="searching this computer…", foreground="#555")
        elif self.ts:
            t = self.ts
            hw = (f"GPU: {t.get('gpu')} ({t.get('vram_gb')} GB)" if t.get("cuda")
                  else "no usable GPU, so it will run on the CPU (slower)")
            ver = f"{t.get('version')}"
            if t.get("version") != C.REFERENCE_TS_VERSION:
                ver += f" (reference version: {C.REFERENCE_TS_VERSION})"
            extra = "" if t.get("weights_ready") is not False else "  ·  model files download on first use"
            self.lbl_ts.configure(text=f"{ver}  ·  {hw}{extra}", foreground="#1e7b34")
        else:
            self.lbl_ts.configure(text="not found. Click the button on the right to install or locate it.",
                                  foreground="#b03a2e")

    def open_ts_dialog(self):
        if self.ts_dialog is not None and self.ts_dialog.winfo_exists():
            self.ts_dialog.lift()
            return
        self.ts_dialog = TSDialog(self)

    # 1. cases
    def _build_cases_tab(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="  1. Cases  ")
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")

        r = ttk.Frame(f)
        r.pack(fill=tk.X)
        ttk.Label(r, text="TotalSegmentator:", font=bold).pack(side=tk.LEFT)
        self.lbl_ts = ttk.Label(r, text="searching this computer…")
        self.lbl_ts.pack(side=tk.LEFT, padx=6)
        ttk.Button(r, text="TotalSegmentator…", command=self.open_ts_dialog).pack(side=tk.RIGHT)
        self.device_var = tk.StringVar(value=self.settings.get("device", "Auto"))
        cb = ttk.Combobox(r, values=list(DEVICES), textvariable=self.device_var, state="readonly", width=6)
        cb.pack(side=tk.RIGHT, padx=(2, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self._device_changed())
        ttk.Label(r, text="Run on:").pack(side=tk.RIGHT)

        r = ttk.Frame(f)
        r.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(r, text="Case folder:", font=bold).pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value=self.settings.get("case_folder", ""))
        ttk.Entry(r, textvariable=self.folder_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.btn_choose = ttk.Button(r, text="Choose…", command=self.choose_folder)
        self.btn_choose.pack(side=tk.LEFT, padx=2)
        self.btn_rescan = ttk.Button(r, text="Rescan", command=lambda: self.scan(force=True))
        self.btn_rescan.pack(side=tk.LEFT, padx=2)
        ttk.Label(f, foreground="#555", padding=(0, 2), text=(
            "One subfolder per case, with that patient's DICOM files anywhere inside it. "
            "Nothing in this folder is changed.")).pack(anchor=tk.W)

        tf = ttk.Frame(f)
        tf.pack(fill=tk.BOTH, expand=True, pady=6)
        cols = ("check", "folder", "case", "series", "slices", "thick", "con", "status", "vol", "a1", "a2", "notes")
        tv = ttk.Treeview(tf, columns=cols, show="headings", height=14, selectmode="browse")
        for c, t, w in (("check", "", 30), ("folder", "Case folder", 150), ("case", "Code", 50), ("series", "Series", 150),
                        ("slices", "Slices", 45), ("thick", "mm", 40), ("con", "Contrast", 55), ("status", "Status", 150),
                        ("vol", "TS mL", 50), ("a1", "AUTH1", 70), ("a2", "AUTH2", 70), ("notes", "Notes", 120)):
            tv.heading(c, text=t)
            tv.column(c, width=px(self, w), anchor=tk.W, stretch=c in ("folder", "series", "notes"))
        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tv.tag_configure("segmented", foreground="#1e7b34")
        tv.tag_configure("active", foreground="#1f4e79")
        tv.tag_configure("problem", foreground="#b03a2e")
        tv.tag_configure("disabled", foreground="#888")
        tv.bind("<Button-1>", self._tv_click)
        tv.bind("<space>", self._tv_space)
        self.tv_cases = tv

        r = ttk.Frame(f)
        r.pack(fill=tk.X)
        self.btn_segment = ttk.Button(r, text="▶ Segment ticked cases", command=self.segment_checked)
        self.btn_segment.pack(side=tk.LEFT, padx=2)
        ttk.Button(r, text="Tick all remaining", command=self.check_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(r, text="Untick all", command=self.check_none).pack(side=tk.LEFT, padx=2)
        ttk.Button(r, text="Change series…", command=self.change_series).pack(side=tk.LEFT, padx=(14, 2))
        ttk.Button(r, text="Delete case…", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(r, text="Workspace folder", command=lambda: self._open_path(C.WORK_DIR)).pack(side=tk.RIGHT, padx=2)

        r = ttk.Frame(f)
        r.pack(fill=tk.X, pady=(6, 0))
        self.btn_stop = ttk.Button(r, text="Stop", command=self.stop_job, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT)
        self.lbl_job = ttk.Label(r, text="", foreground="#1f4e79")
        self.lbl_job.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pb_cases = ttk.Progressbar(f, mode="determinate", maximum=100)
        self.pb_cases.pack(fill=tk.X, pady=(4, 0))
        self.log_cases = ScrolledText(f, height=5, state=tk.DISABLED, font=("Consolas", 9))
        self.log_cases.pack(fill=tk.X, pady=(6, 0))

    def _device_changed(self):
        self.settings["device"] = self.device_var.get()
        self._save_settings()

    # case table
    def _job_running(self):
        return bool(self.job.get("running") and joblock.pid_alive(int(self.job.get("pid", 0))))

    def _collect_rows(self):
        root = self.settings.get("case_folder")
        by_uid = cases.cases_by_series()
        running = self._job_running()
        states = dict(self.job.get("items", {}))
        if running:
            queued = cases.read_json(C.JOB_QUEUE, None) or {}
            for it in queued.get("items", []):
                states.setdefault(it["key"], {"state": "queued"})
        rows, used = [], set()
        for sc in (folder_scan.load_rows(root) if root else []):
            uid, cid = sc.get("series_uid"), None
            for s in sc["series"]:
                if s["series_uid"] in by_uid:
                    uid, cid = s["series_uid"], by_uid[s["series_uid"]]
                    break
            if cid:
                used.add(cid)
            ser = next((s for s in sc["series"] if s["series_uid"] == uid), None)
            rows.append(self._make_row(sc["path"], sc["name"], cid, uid, ser, sc, states.get(sc["path"]), running))
        for cid in cases.list_cases():
            if cid not in used:
                key = f"case:{cid}"
                rows.append(self._make_row(key, "", cid, None, None, None, states.get(key), running))
        return rows

    def _make_row(self, key, name, cid, uid, ser, scan, state, running):
        meta = cases.read_json(cases.case_dir(cid) / C.META_FILE, {}) if cid else {}
        ts_info = cases.read_json(cases.case_dir(cid) / C.TS_INFO_FILE, None) if cid else None
        done = bool(cid) and cases.ts_done(cid)
        job_state = (state or {}).get("state")
        active = running and job_state in ("queued", "importing", "running")
        notes = list(meta.get("warnings", []))
        if scan and scan.get("n_patients", 0) > 1:
            notes.append(f"{scan['n_patients']} different patients in this folder")
        elif scan and scan.get("n_studies", 0) > 1:
            notes.append(f"{scan['n_studies']} studies in this folder")
        if ts_info and ts_info.get("device") == "cpu":
            notes.append("segmented on CPU")
        tag = ""
        if done:
            status, tag = "Segmented", "segmented"
        elif active:
            status = {"queued": "Queued", "importing": "Importing…", "running": "Segmenting…"}[job_state]
            tag = "active"
        elif job_state == "failed":
            status, tag = f"Failed: {state.get('error', '')}", "problem"
        elif scan and scan.get("error"):
            status, tag = "Could not read the folder", "problem"
        elif scan and not scan["series"]:
            status, tag = "No DICOM files found", "problem"
        elif scan and scan.get("needs_review") and not cid:
            status, tag = "Check the series before use", "problem"
        elif cid:
            status = "Imported, not segmented"
        else:
            status = "Not segmented"
        can_check = not done and not active and (bool(cid) or (bool(scan) and bool(scan["series"])))
        return {
            "key": key, "name": name or "(imported earlier)", "case": cid, "series_uid": uid, "scan": scan,
            "folder": scan["path"] if scan else None, "can_check": can_check, "tag": tag if can_check or done or active else "disabled",
            "values": (
                "☑" if key in self.checked else ("☐" if can_check else ""),
                name or "(imported earlier)", cid or "",
                meta.get("series_desc", (ser or {}).get("desc", "")),
                (meta.get("size_xyz") or ["", "", (ser or {}).get("n", "")])[2],
                meta.get("slice_thickness_mm") or (ser or {}).get("thickness") or "",
                meta.get("contrast_hint", (ser or {}).get("contrast", "")),
                status,
                f"{ts_info['volume_ml']:.0f}" if ts_info else "",
                READER_STATUS_TEXT.get(cases.reader_status(cid, "AUTH1").get("status"), "") if done else "",
                READER_STATUS_TEXT.get(cases.reader_status(cid, "AUTH2").get("status"), "") if done else "",
                "; ".join(notes),
            ),
        }

    def _refresh_cases(self):
        tv = self.tv_cases
        keep = {self._iid_key.get(i) for i in tv.selection()}
        top = tv.yview()[0]
        self.rows = self._collect_rows()
        self.checked &= {r["key"] for r in self.rows if r["can_check"]}
        tv.delete(*tv.get_children())
        self._iid_key = {}
        for i, row in enumerate(self.rows):
            iid = f"r{i}"
            self._iid_key[iid] = row["key"]
            values = list(row["values"])
            values[0] = "☑" if row["key"] in self.checked else ("☐" if row["can_check"] else "")
            tv.insert("", tk.END, iid=iid, values=values, tags=(row["tag"],) if row["tag"] else ())
            if row["key"] in keep:
                tv.selection_set(iid)
        tv.yview_moveto(top)

    def _toggle(self, iid):
        key = self._iid_key.get(iid)
        row = next((r for r in self.rows if r["key"] == key), None)
        if not row or not row["can_check"]:
            return
        self.checked.symmetric_difference_update({key})
        self.tv_cases.set(iid, "check", "☑" if key in self.checked else "☐")

    def _tv_click(self, e):
        tv = self.tv_cases
        if tv.identify_region(e.x, e.y) == "cell" and tv.identify_column(e.x) == "#1":
            iid = tv.identify_row(e.y)
            if iid:
                self._toggle(iid)
                return "break"
        return None

    def _tv_space(self, _e):
        for iid in self.tv_cases.selection():
            self._toggle(iid)
        return "break"

    def check_all(self):
        self.checked = {r["key"] for r in self.rows if r["can_check"]}
        self._refresh_cases()

    def check_none(self):
        self.checked = set()
        self._refresh_cases()

    def _selected_row(self):
        sel = self.tv_cases.selection()
        key = self._iid_key.get(sel[0]) if sel else None
        return next((r for r in self.rows if r["key"] == key), None)

    # case folder
    def choose_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Choose the folder that holds the case subfolders",
                                         initialdir=self.settings.get("case_folder") or None)
        if not folder:
            return
        folder = str(Path(folder))
        self.settings["case_folder"] = folder
        self._save_settings()
        self.folder_var.set(folder)
        self.scan()

    def _startup_scan(self):
        root = self.settings.get("case_folder")
        if root and Path(root).is_dir():
            self.scan()
        elif root:
            self.log(f"The case folder is not available right now: {root}")

    def scan(self, force = False):
        root = self.settings.get("case_folder")
        if not root:
            messagebox.showinfo("Case folder", "Choose the case folder first.", parent=self)
            return
        if self.scanning:
            return
        self.scanning = True
        self.btn_choose.configure(state=tk.DISABLED)
        self.btn_rescan.configure(state=tk.DISABLED)

        def progress(i, n):
            self.ui(lambda: (self.lbl_job.configure(text=f"Reading case folder {i} of {n}…"),
                             self.pb_cases.configure(value=100 * (i - 1) / n)))

        def work():
            error = None
            try:
                count = folder_scan.scan_root(root, force, progress, lambda: self.ui(self._refresh_cases))
            except Exception as exc:
                count, error = 0, f"{type(exc).__name__}: {exc}"

            def done():
                self.scanning = False
                self.btn_choose.configure(state=tk.NORMAL)
                self.btn_rescan.configure(state=tk.NORMAL)
                self.pb_cases.configure(value=0)
                self.lbl_job.configure(text="")
                if error:
                    messagebox.showerror("Could not read the folder", error, parent=self)
                elif count:
                    self.log(f"Read {count} case folder(s).")
                self._refresh_cases()
            self.ui(done)

        threading.Thread(target=work, daemon=True).start()

    def change_series(self):
        row = self._selected_row()
        if not row:
            messagebox.showinfo("Change series", "Select a case in the table first.", parent=self)
            return
        if row["case"]:
            messagebox.showinfo("Change series", f"{row['case']} has already been imported. To use another series, "
                                "delete the case first.", parent=self)
            return
        if row["scan"] and row["scan"]["series"]:
            self._ask_series(row)
            self._refresh_cases()

    def _ask_series(self, row):
        dlg = SeriesDialog(self, f"Series: {row['name']}", row["scan"]["series"], row["series_uid"])
        self.wait_window(dlg)
        if dlg.result:
            folder_scan.set_series(self.settings.get("case_folder"), row["key"], dlg.result)
            row["series_uid"] = dlg.result
            return True
        return False

    def delete_selected(self):
        row = self._selected_row()
        if not row or not row["case"]:
            messagebox.showinfo("Delete case", "Select an imported case in the table first.", parent=self)
            return
        cid = row["case"]
        if self._job_running():
            messagebox.showinfo("Delete case", "Please wait until the segmentation job has finished.", parent=self)
            return
        if any(k[0] == cid for k in self.editors):
            messagebox.showwarning("Open", "This case is open in a correction window.", parent=self)
            return
        if messagebox.askyesno("Delete case", f"Permanently delete {cid} and all its masks (TS, AUTH1, AUTH2)?\n"
                               "The original DICOM files are not touched.", icon=messagebox.WARNING, parent=self):
            cases.delete_case(cid)
            self.log(f"{cid} deleted.")
            self.refresh()

    # segmentation job
    def segment_checked(self):
        rows = [r for r in self.rows if r["key"] in self.checked and r["can_check"]]
        if not rows:
            messagebox.showinfo("Segment", "Tick at least one case in the first column.", parent=self)
            return
        if self.ts_state == "searching":
            messagebox.showinfo("TotalSegmentator", "Still looking for TotalSegmentator. Try again in a moment.", parent=self)
            return
        if not self.ts:
            self.open_ts_dialog()
            return
        items = []
        for row in rows:
            if not row["case"] and row["scan"] and row["scan"].get("needs_review"):
                if not self._ask_series(row):
                    self.checked.discard(row["key"])
                    continue
            if row["case"] or row["series_uid"]:
                items.append({"key": row["key"], "folder": row["folder"], "series_uid": row["series_uid"],
                              "case": row["case"]})
        if not items:
            self._refresh_cases()
            return
        running = self._job_running() or joblock.holder() is not None or \
            (self._job_pid and joblock.pid_alive(self._job_pid))
        if running:
            q = cases.read_json(C.JOB_QUEUE, None)
            if q is None:
                messagebox.showinfo("Segment", "The job is finishing. Try again in a moment.", parent=self)
                return
            known = {it["key"] for it in q["items"]}
            q["items"].extend(it for it in items if it["key"] not in known)
            cases.write_json(C.JOB_QUEUE, q)
            self.log(f"{len(items)} case(s) added to the running job.")
        else:
            cases.write_json(C.JOB_QUEUE, {"python": self.ts["python"], "device": DEVICES[self.device_var.get()],
                                           "items": items})
            self._launch_job()
        self.checked -= {it["key"] for it in items}
        self._poll_job(schedule=False)
        self._refresh_cases()

    def _launch_job(self):
        C.ensure_dirs()
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        out = open(C.LOG_DIR / f"job_{stamp}.log", "w", encoding="utf-8")
        err = open(C.LOG_DIR / f"job_{stamp}_console.log", "w", encoding="utf-8")
        # A process independent of this window: new process group, no console window and,
        # where allowed, out of the window's job object
        base = CREATE_NO_WINDOW | 0x00000200  # CREATE_NEW_PROCESS_GROUP
        kw = dict(cwd=str(C.SRC_DIR), stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=worker_env(), close_fds=True)
        args = [WORKER_PY, str(C.SRC_DIR / "seg_job.py")]
        try:
            try:
                p = subprocess.Popen(args, creationflags=base | 0x01000000, **kw)  # CREATE_BREAKAWAY_FROM_JOB
            except OSError:
                p = subprocess.Popen(args, creationflags=base, **kw)
        finally:
            out.close()
            err.close()
        self._job_pid = p.pid
        self.log("Segmentation started. It continues in the background if you close this window.")

    def stop_job(self):
        ans = messagebox.askyesnocancel(
            "Stop",
            "Yes: finish the current case, then stop.\n"
            "No: stop now and discard the current case's progress.\n\n"
            "Cases not reached stay unsegmented; you can tick them again at any time.", parent=self)
        if ans is None:
            return
        try:
            C.JOB_STOP.write_text("after" if ans else "now", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not stop", str(exc), parent=self)

    def _average_seconds(self):
        vals = []
        for cid in cases.list_cases():
            info = cases.read_json(cases.case_dir(cid) / C.TS_INFO_FILE, None)
            if info and info.get("seconds"):
                vals.append(float(info["seconds"]))
        return sum(vals) / len(vals) if vals else None

    def _poll_job(self, schedule = True):
        try:
            st = cases.read_json(C.JOB_STATUS, None) or {}
            self.job = st
            running = self._job_running()
            text, pct = "", 0.0
            if running:
                i, n = int(st.get("i", 0)), int(st.get("n", 0))
                pct = 100 * (max(i - 1, 0) + st.get("case_percent", 0) / 100) / max(n, 1)
                text = f"Case {i} of {n}"
                if st.get("current_case"):
                    text += f" ({st['current_case']})"
                text += f": {st.get('message', '')}"
                if st.get("case_started"):
                    elapsed = (datetime.now() - datetime.fromisoformat(st["case_started"])).total_seconds()
                    text += f", {fmt_duration(elapsed)}"
                avg = self._average_seconds()
                if avg:
                    text += f"   ·   about {fmt_duration(avg)} per case on this computer"
                text += "   ·   continues if you close this window"
                try:
                    stop = C.JOB_STOP.read_text(encoding="utf-8").strip()
                except OSError:
                    stop = ""
                if stop:
                    text += "   ·   stopping…"
            elif st.get("running"):
                text = f"The segmentation job ended unexpectedly (restart or crash?). Details: {C.LOG_DIR}"
            elif st.get("phase") in ("finished", "stopped", "error"):
                when = (st.get("updated") or "").replace("T", " ")
                text = f"Last job ({when}): {st.get('phase')}. {st.get('message', '')}"
            elif self._job_pid and joblock.pid_alive(self._job_pid):
                text = "Starting the segmentation job…"
            if not self.scanning:
                self.lbl_job.configure(text=text)
                self.pb_cases.configure(value=pct)
            self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)
            for item_key, item in st.get("items", {}).items():
                mark = (st.get("started"), item_key)
                if item.get("state") == "failed" and mark not in self._failures_logged:
                    self._failures_logged.add(mark)
                    self.log(f"{item.get('case') or Path(item_key).name} failed: {item.get('error', '')}")
            key = st.get("updated")
            if key != self._job_key:
                first = self._job_key is None
                self._job_key = key
                if not first:
                    self.refresh()
        except Exception as exc:
            print(f"poll: {exc}", file=sys.stderr)
        if schedule:
            self.after(2000, self._poll_job)

    # 2. correction
    def _build_reader_tab(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="  2. Correction  ")
        top = ttk.Frame(f)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Reader:").pack(side=tk.LEFT)
        saved = self.settings.get("reader")
        self.reader_var = tk.StringVar(value=saved if saved in C.READERS else "")
        cb = ttk.Combobox(top, values=C.READERS, textvariable=self.reader_var, state="readonly", width=8)
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._reader_changed())
        ttk.Label(top, foreground="#555", text=(
            "   Each reader sees only the TS mask and their own correction. The case order differs per reader."
        )).pack(side=tk.LEFT)
        cols = ("order", "case", "status", "time", "saved")
        tv = ttk.Treeview(f, columns=cols, show="headings", height=18, selectmode="browse")
        for c, t, w in (("order", "#", 50), ("case", "Case", 80), ("status", "Status", 120),
                        ("time", "Active time (min)", 120), ("saved", "Last saved", 170)):
            tv.heading(c, text=t)
            tv.column(c, width=px(self, w), anchor=tk.W)
        tv.pack(fill=tk.BOTH, expand=True, pady=6)
        tv.bind("<Double-1>", lambda e: self.open_selected())
        tv.tag_configure(C.DONE, foreground="#1e7b34")
        tv.tag_configure(C.IN_PROGRESS, foreground="#b9770e")
        self.tv_reader = tv
        bf = ttk.Frame(f)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="▶ Open next pending case", command=self.open_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Open selected case", command=self.open_selected).pack(side=tk.LEFT, padx=2)
        self.reader_info = ttk.Label(f, padding=(0, 6))
        self.reader_info.pack(fill=tk.X)

    def _reader_changed(self):
        self.settings["reader"] = self.reader_var.get()
        self._save_settings()
        self._refresh_reader()

    def _refresh_reader(self):
        tv = self.tv_reader
        keep = tv.selection()
        tv.delete(*tv.get_children())
        r = self.reader_var.get()
        if r not in C.READERS:
            self.reader_info.configure(text="Choose the reader first (AUTH1 or AUTH2).")
            return
        all_cases = cases.list_cases()
        ready = [c for c in all_cases if cases.ts_done(c)]
        done = 0
        for i, cid in enumerate(cases.reader_order(r, ready), 1):
            st = cases.reader_status(cid, r)
            s = st.get("status", C.PENDING)
            done += s == C.DONE
            tv.insert("", tk.END, iid=cid, tags=(s,), values=(
                i, cid, READER_STATUS_TEXT.get(s, s), fmt_min(st.get("active_seconds", 0)) if s != C.PENDING else "",
                (st.get("last_saved") or "").replace("T", " ")))
        waiting = len(all_cases) - len(ready)
        self.reader_info.configure(text=f"{r}: {done} of {len(ready)} cases done."
                                        + (f"   ·   {waiting} imported case(s) still have no segmentation."
                                           if waiting else ""))
        keep = [k for k in keep if tv.exists(k)]
        if keep:
            tv.selection_set(keep)

    def _open_editor(self, cid, reader):
        key = (cid, reader)
        if key in self.editors:
            try:
                self.editors[key].lift()
                self.editors[key].focus_force()
                return
            except tk.TclError:
                self.editors.pop(key, None)
        from editor import CorrectionEditor
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            ed = CorrectionEditor(self, cid, reader, on_closed=self._editor_closed)
        except Exception as exc:
            messagebox.showerror("Could not open", f"{cid}: {exc}", parent=self)
            return
        finally:
            self.config(cursor="")
        self.editors[key] = ed
        ed.focus_force()

    def _editor_closed(self, cid, reader):
        self.editors.pop((cid, reader), None)
        self.refresh()

    def open_selected(self):
        r = self.reader_var.get()
        sel = self.tv_reader.selection()
        if r in C.READERS and sel:
            self._open_editor(sel[0], r)

    def open_next(self):
        r = self.reader_var.get()
        if r not in C.READERS:
            messagebox.showinfo("Reader", "Choose the reader first.", parent=self)
            return
        ready = [c for c in cases.list_cases() if cases.ts_done(c)]
        for cid in cases.reader_order(r, ready):
            if cases.reader_status(cid, r).get("status") != C.DONE:
                self.tv_reader.selection_set(cid)
                self.tv_reader.see(cid)
                self._open_editor(cid, r)
                return
        messagebox.showinfo("All done", f"No pending cases for {r}.", parent=self)

    # 3. analysis
    def _build_analysis_tab(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="  3. Analysis  ")
        self.an_info = ttk.Label(f, justify=tk.LEFT)
        self.an_info.pack(fill=tk.X)
        ttk.Label(f, justify=tk.LEFT, foreground="#555", padding=(0, 6), text=(
            "Saved for every case done by both readers: liver volume (mL) and mean density (HU) of TS, AUTH1 and AUTH2.\n"
            "Calculated for TS vs AUTH1, TS vs AUTH2 and AUTH1 vs AUTH2:\n"
            "    ICC(2,1) of volume and of mean density (two-way random, absolute agreement, single measurement) with 95% CI\n"
            "    Dice and IoU (per case, and mean ± SD, median, minimum), plus HD95 and ASSD boundary distances\n"
            "Results are tables (CSV), Bland-Altman plots and an HTML report.")).pack(fill=tk.X)
        bf = ttk.Frame(f)
        bf.pack(fill=tk.X, pady=4)
        self.btn_an = ttk.Button(bf, text="▶ Run analysis", command=self.run_analysis)
        self.btn_an.pack(side=tk.LEFT, padx=2)
        self.btn_report = ttk.Button(bf, text="Open last report", state=tk.DISABLED,
                                     command=lambda: self._open_path(self.last_result / "report.html"))
        self.btn_report.pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Results folder", command=lambda: self._open_path(C.RESULTS_DIR)).pack(side=tk.LEFT, padx=2)
        self.pb_an = ttk.Progressbar(f, mode="determinate", maximum=100)
        self.pb_an.pack(fill=tk.X, pady=4)
        self.log_an = ScrolledText(f, height=14, state=tk.DISABLED, font=("Consolas", 9))
        self.log_an.pack(fill=tk.BOTH, expand=True)

    def _refresh_analysis(self):
        all_cases = cases.list_cases()
        ok = [c for c in all_cases if all(cases.reader_status(c, r).get("status") == C.DONE for r in C.READERS)]
        self.an_info.configure(text=f"Cases done by both readers: {len(ok)} of {len(all_cases)}"
                                    + ("" if len(ok) >= 3 else "   (ICC needs at least 3)"))
        runs = sorted(C.RESULTS_DIR.glob("*/report.html")) if C.RESULTS_DIR.exists() else []
        if runs:
            self.last_result = runs[-1].parent
            self.btn_report.configure(state=tk.NORMAL)

    def run_analysis(self):
        if self.installing:
            messagebox.showinfo("Analysis", "Please wait until the installation has finished.", parent=self)
            return
        self.btn_an.configure(state=tk.DISABLED)
        self.pb_an["value"] = 0
        self.log("Analysis started…", "an")
        res = {}

        def on_line(line):
            parts = line.split("|")
            if parts[0] == "PROG" and len(parts) >= 3:
                self.ui(lambda: self.pb_an.configure(value=float(parts[1])))
            elif parts[0] == "DONE":
                res["out"] = Path("|".join(parts[1:]))
            elif parts[0] == "RESULT":
                self.ui(lambda: self.log("|".join(parts[1:]), "an"))
            elif parts[0] == "FAIL":
                self.ui(lambda: self.log("ERROR: " + "|".join(parts[1:]), "an"))

        def on_exit(rc):
            self.btn_an.configure(state=tk.NORMAL)
            if rc == 0 and "out" in res:
                self.last_result = res["out"]
                self.btn_report.configure(state=tk.NORMAL)
                self.pb_an["value"] = 100
                self.log(f"Finished: {res['out']}", "an")
                self._open_path(res["out"] / "report.html")
            else:
                self.log(f"The analysis did not finish (details: {C.LOG_DIR})", "an")
            self.refresh()

        errlog = C.LOG_DIR / f"analysis_{datetime.now():%Y%m%d_%H%M%S}.log"

        def work():
            with open(errlog, "w", encoding="utf-8", errors="replace") as ef:
                p = subprocess.Popen([WORKER_PY, str(C.SRC_DIR / "analyze.py")], cwd=str(C.SRC_DIR),
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=ef, env=worker_env(),
                                     creationflags=CREATE_NO_WINDOW, text=True, encoding="utf-8", errors="replace",
                                     bufsize=1)
                self.proc = p
                for line in p.stdout:
                    on_line(line.rstrip("\n"))
                rc = p.wait()
            self.proc = None
            self.ui(lambda: on_exit(rc))

        threading.Thread(target=work, daemon=True).start()

    # closing
    def _on_close(self):
        for ed in list(self.editors.values()):
            try:
                ed._on_close()
            except tk.TclError:
                pass
        if self.editors:
            return  # a window cancelled closing
        if self.proc is not None:
            what = "An installation" if self.installing else "An analysis"
            if not messagebox.askyesno("Still working", f"{what} is running. Stop it and exit?", parent=self):
                return
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.destroy()


def _setup_error_log():
    """pythonw has no console: errors go to a file."""
    C.ensure_dirs()
    if sys.stderr is None or not getattr(sys.stderr, "isatty", lambda: False)():
        try:
            sys.stderr = open(C.LOG_DIR / f"app_{datetime.now():%Y%m%d}.log", "a", encoding="utf-8", buffering=1)
            if sys.stdout is None:
                sys.stdout = sys.stderr
        except OSError:
            pass


def main():
    _setup_error_log()
    enable_dpi_awareness()
    app = App()

    def report(exc, val, tb):
        import traceback
        traceback.print_exception(exc, val, tb, file=sys.stderr)
        try:
            messagebox.showerror("Unexpected error", f"{exc.__name__}: {val}\n\nDetails: {C.LOG_DIR}", parent=app)
        except Exception:
            pass

    app.report_callback_exception = report
    app.mainloop()


if __name__ == "__main__":
    main()
