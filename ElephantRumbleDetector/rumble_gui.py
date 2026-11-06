#!/usr/bin/env python
"""
The window of the elephant rumble detector: pick the recordings and a results folder, press
Start. It runs `rumble_detector.py run` in the background (restarting it if it gets stuck),
shows its progress and messages, and can change the cut-off of finished results.

Opened by "2 - Find rumbles.bat". Uses only tkinter, which comes with Python.
"""
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rumble_detector as rd  # noqa: E402 - imports nothing heavy until a command runs

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

TITLE = "Elephant Rumble Detector"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no black window for the worker
PAD = 8
# the detector runs in python.exe (hidden), not pythonw.exe, so that its printed lines reach the window
PYTHON = os.environ.get("RUMBLE_WORKER_PYTHON") or (
    os.path.join(os.path.dirname(sys.executable), "python.exe")
    if os.path.basename(sys.executable).lower() == "pythonw.exe" else sys.executable)


def gpu_name():
    """The NVIDIA card's name, or '' (no card, no driver, or the installer chose the processor)."""
    if os.path.exists(rd.USE_CPU_FILE):
        return ""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20, creationflags=NO_WINDOW)
        return r.stdout.strip().splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        return ""


def open_folder(path):
    if not os.path.isdir(path):
        return
    if os.name == "nt":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Worker:
    """Runs one detector command in the background and passes its printed lines to the window.
    `run` is started again when it stops itself because it was stuck (exit code 3)."""

    def __init__(self, cmd, lines, restart=True):
        self.cmd, self.lines, self.restart = cmd, lines, restart
        self.proc, self.stopping, self.code = None, False, None
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def work(self):
        restarts = 0
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            while True:
                self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                             errors="replace", env=env, creationflags=NO_WINDOW, cwd=HERE)
                for line in self.proc.stdout:
                    self.lines.put(("line", line.rstrip("\n")))
                code = self.proc.wait()
                if code == rd.STALLED and self.restart and not self.stopping and restarts < 50:
                    restarts += 1
                    self.lines.put(("line", f"{time.strftime('%H:%M:%S')} starting again where it stopped..."))
                    continue
                self.code = code
                break
        except Exception as e:  # noqa: BLE001 - e.g. Python missing: show it, don't vanish
            self.lines.put(("line", f"Could not start the detector: {e}"))
            self.code = -1
        self.lines.put(("done", self.code))

    def stop(self):
        self.stopping = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()  # results are written atomically: a finished recording is never half-written


class App:
    def __init__(self, root, given=None):
        self.root = root
        self.lines = queue.Queue()
        self.worker = None
        self.total = self.done = 0
        self.out_edited = False
        root.title(TITLE)
        root.minsize(720, 560)
        try:
            root.tk.call("tk", "scaling", max(1.0, root.winfo_fpixels("1i") / 72))
        except tk.TclError:
            pass
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 6))
        style.configure("Head.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        style.configure("Status.TLabel", font=("Segoe UI", 10))

        main = ttk.Frame(root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # 1. recordings
        ttk.Label(main, text="1.  Recordings", style="Head.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="A whole deployment (a folder with one folder per site), one site folder, "
                             "or one recording.", style="Hint.TLabel").grid(row=1, column=0, sticky="w")
        f1 = ttk.Frame(main)
        f1.grid(row=2, column=0, sticky="ew", pady=(2, PAD))
        f1.columnconfigure(0, weight=1)
        self.inp = tk.StringVar(value=given or "")
        self.inp.trace_add("write", lambda *_: self.input_changed())
        ttk.Entry(f1, textvariable=self.inp).grid(row=0, column=0, sticky="ew")
        ttk.Button(f1, text="Choose folder...", command=self.pick_folder).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(f1, text="Choose one file...", command=self.pick_file).grid(row=0, column=2, padx=(6, 0))

        # 2. results
        ttk.Label(main, text="2.  Save the results in", style="Head.TLabel").grid(row=3, column=0, sticky="w")
        ttk.Label(main, text="Never inside the recordings folder. Using the same folder again carries on "
                             "where it stopped.", style="Hint.TLabel").grid(row=4, column=0, sticky="w")
        f2 = ttk.Frame(main)
        f2.grid(row=5, column=0, sticky="ew", pady=(2, PAD))
        f2.columnconfigure(0, weight=1)
        self.out = tk.StringVar()
        e = ttk.Entry(f2, textvariable=self.out)
        e.grid(row=0, column=0, sticky="ew")
        e.bind("<Key>", lambda _e: setattr(self, "out_edited", True))
        ttk.Button(f2, text="Choose folder...", command=self.pick_out).grid(row=0, column=1, padx=(6, 0))

        # 3. this computer
        ttk.Label(main, text="3.  This computer", style="Head.TLabel").grid(row=6, column=0, sticky="w")
        f3 = ttk.Frame(main)
        f3.grid(row=7, column=0, sticky="ew", pady=(2, PAD))
        self.hw = ttk.Label(f3, text="Checking for a graphics card...")
        self.hw.grid(row=0, column=0, sticky="w")
        self.cores_frame = ttk.Frame(f3)
        n = os.cpu_count() or 1
        ttk.Label(self.cores_frame, text=f"Processor cores to use (of {n}):").pack(side="left")
        self.cores = tk.IntVar(value=n)
        ttk.Spinbox(self.cores_frame, from_=1, to=n, width=5, textvariable=self.cores).pack(side="left", padx=6)
        ttk.Label(self.cores_frame, text="Fewer keeps the computer free for other work, but is slower.",
                  style="Hint.TLabel").pack(side="left")
        self.gpu = ""
        threading.Thread(target=self.check_gpu, daemon=True).start()

        # buttons
        fb = ttk.Frame(main)
        fb.grid(row=8, column=0, sticky="ew", pady=(4, PAD))
        self.start_b = ttk.Button(fb, text="Start", style="Big.TButton", command=self.start)
        self.start_b.pack(side="left")
        self.stop_b = ttk.Button(fb, text="Stop", command=self.stop, state="disabled")
        self.stop_b.pack(side="left", padx=6)
        self.open_b = ttk.Button(fb, text="Open results folder", command=lambda: open_folder(self.out.get()),
                                 state="disabled")
        self.open_b.pack(side="left")
        ttk.Button(fb, text="Change the cut-off of finished results...", command=self.cutoff_dialog).pack(side="right")

        # progress
        self.bar = ttk.Progressbar(main, mode="determinate", maximum=1)
        self.bar.grid(row=9, column=0, sticky="ew")
        self.status = ttk.Label(main, text="Choose the recordings, then press Start.", style="Status.TLabel")
        self.status.grid(row=10, column=0, sticky="w", pady=(4, 4))

        # messages
        fl = ttk.Frame(main)
        fl.grid(row=11, column=0, sticky="nsew")
        main.rowconfigure(11, weight=1)
        fl.columnconfigure(0, weight=1)
        fl.rowconfigure(0, weight=1)
        self.log = tk.Text(fl, height=12, wrap="word", font=("Consolas", 9), state="disabled",
                           background="#fbfbfb", relief="solid", borderwidth=1)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(fl, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)
        self.log.tag_configure("warn", foreground="#b35c00")
        self.log.tag_configure("bad", foreground="#c00000")
        self.log.tag_configure("good", foreground="#1a7f37")

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)
        if given:
            self.input_changed()

    # -- choosing ------------------------------------------------------------------------------
    def pick_folder(self):
        d = filedialog.askdirectory(title="Choose the deployment or site folder")
        if d:
            self.inp.set(os.path.normpath(d))

    def pick_file(self):
        f = filedialog.askopenfilename(title="Choose one recording",
                                       filetypes=[("Sound files", " ".join("*" + e for e in rd.AUDIO_EXT)),
                                                  ("All files", "*.*")])
        if f:
            self.inp.set(os.path.normpath(f))

    def pick_out(self):
        d = filedialog.askdirectory(title="Choose where to save the results")
        if d:
            self.out.set(os.path.normpath(d))
            self.out_edited = True

    def input_changed(self):
        inp = rd.clean_path(self.inp.get())
        if inp and not self.out_edited:
            self.out.set(rd.default_out(inp))
        self.open_b.configure(state="normal" if os.path.isdir(self.out.get()) else "disabled")

    def check_gpu(self):
        self.lines.put(("gpu", gpu_name()))  # tkinter may only be touched from the main thread

    def show_hardware(self):
        if self.gpu:
            self.hw.configure(text=f"Graphics card: {self.gpu}. It does the heavy work, so this will be fast.")
        else:
            why = ("the installer chose the processor on this computer" if os.path.exists(rd.USE_CPU_FILE)
                   else "no NVIDIA graphics card found")
            self.hw.configure(text=f"Runs on the processor ({why}). This works, but is slower than a graphics card.")
            self.cores_frame.grid(row=1, column=0, sticky="w", pady=(4, 0))

    # -- running -------------------------------------------------------------------------------
    def check_inputs(self):
        inp = rd.clean_path(self.inp.get())
        out = rd.clean_path(self.out.get())
        if not inp:
            messagebox.showwarning(TITLE, "Choose the recordings first (a folder or one sound file).")
            return None
        if not (os.path.isdir(inp) or (os.path.isfile(inp) and rd.is_audio(inp))):
            messagebox.showwarning(TITLE, f"Can't find a folder or sound file at:\n\n{inp}\n\n"
                                          "Check the spelling, and that the drive (for example L:) is connected.")
            return None
        if not out:
            messagebox.showwarning(TITLE, "Choose where to save the results.")
            return None
        out = os.path.abspath(out)
        if os.path.isdir(inp) and rd.is_under(out, inp):
            messagebox.showwarning(TITLE, "The results can't go inside the recordings folder. Choose another place.")
            return None
        return inp, out

    def start(self):
        got = self.check_inputs()
        if not got:
            return
        inp, out = got
        self.out.set(out)
        cmd = [PYTHON, os.path.join(HERE, "rumble_detector.py"), "run", inp, "--out", out]
        if not self.gpu:
            try:
                c = int(self.cores.get())
                if 1 <= c < (os.cpu_count() or 1):
                    cmd += ["--cores", str(c)]
            except (tk.TclError, ValueError):
                pass
        cmd += os.environ.get("RUMBLE_TEST_ARGS", "").split()  # only for the tests
        self.clear_log()
        self.total = self.done = 0
        self.bar.configure(value=0, maximum=1)
        self.set_status("Starting... (looking for the recordings)")
        self.set_running(True)
        self.worker = Worker(cmd, self.lines)

    def stop(self):
        if self.worker and messagebox.askyesno(TITLE, "Stop the detector?\n\nFinished recordings are kept. "
                                                      "Press Start again later with the same folders to carry on."):
            self.worker.stop()
            self.set_status("Stopping...")

    def set_running(self, running):
        for w in (self.start_b,):
            w.configure(state="disabled" if running else "normal")
        self.stop_b.configure(state="normal" if running else "disabled")

    def poll(self):
        try:
            while True:
                kind, value = self.lines.get_nowait()
                if kind == "line":
                    self.add_line(value)
                elif kind == "gpu":
                    self.gpu = value
                    self.show_hardware()
                else:
                    self.finished(value)
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def add_line(self, line):
        m = re.search(r"(\d+) recordings in \d+ folder\(s\): (\d+) already done, (\d+) to do", line)
        if m:
            self.total, self.done = int(m.group(3)), 0
            self.bar.configure(maximum=max(1, self.total), value=0)
            already = int(m.group(2))
            self.set_status(f"{self.total} recordings to do" + (f" ({already} already done earlier)" if already else ""))
            self.open_b.configure(state="normal")
        m = re.search(r"\[(\d+)/(\d+)\] (.*)", line)
        if m:
            self.done, self.total = int(m.group(1)), int(m.group(2))
            self.bar.configure(maximum=max(1, self.total), value=self.done)
            left = re.search(r"about (.+) left", line)
            self.set_status(f"{self.done} of {self.total} recordings done" + (f", about {left.group(1)} left" if left else ""))
        if "hours of this recording analysed" in line and self.total:
            self.set_status(f"{self.done} of {self.total} recordings done; working on the next one"
                            f" ({line.split('analysed')[0].strip().split(' ', 1)[1]} analysed)")
        tag = ("bad" if re.search(r"SKIPPED|STOPPED|Traceback|Error|Can't find|could not", line)
               else "warn" if "WARNING" in line else "good" if re.search(r"all \d+ recordings have", line) else None)
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def finished(self, code):
        self.worker = None
        self.set_running(False)
        out = self.out.get()
        self.open_b.configure(state="normal" if os.path.isdir(out) else "disabled")
        if code == 0:
            self.bar.configure(value=self.bar.cget("maximum"))
            failed = os.path.exists(os.path.join(out, "failed.txt"))
            self.set_status("Finished." + (" Some recordings could not be read: see failed.txt." if failed else ""))
            if messagebox.askyesno(TITLE, "Finished.\n\n" + ("Some recordings could not be read; the reasons are "
                                   "in failed.txt.\n\n" if failed else "") +
                                   "Open the results folder? Start with 'ABOUT THESE RESULTS.txt' and summary.md."):
                open_folder(out)
        elif code == rd.DRIVE_GONE:
            self.set_status("Stopped: the recordings can't be reached. Connect the drive and press Start again.")
            messagebox.showwarning(TITLE, "The recordings stopped being reachable (the drive may have disconnected).\n\n"
                                          "Connect it and press Start again: it carries on where it stopped.")
        elif self.status.cget("text") == "Stopping...":
            self.set_status("Stopped. Press Start with the same folders to carry on.")
        else:
            self.set_status("It stopped with a problem: see the messages below.")
            messagebox.showerror(TITLE, "The detector stopped with a problem (the messages are in the window and in "
                                        "run_log.txt in the results folder).\n\nPressing Start again carries on where "
                                        "it stopped. If it keeps happening, send run_log.txt to Amogh.")

    def set_status(self, text):
        self.status.configure(text=text)

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def close(self):
        if self.worker:
            if not messagebox.askyesno(TITLE, "The detector is still running. Stop it and close?\n\n"
                                              "Finished recordings are kept."):
                return
            self.worker.stop()
        self.root.destroy()

    # -- changing the cut-off ------------------------------------------------------------------
    def cutoff_dialog(self):
        CutoffDialog(self)


def load_test_results():
    """{cut-off: {column: value}} from model\\test_results.csv (run 0 on ELP's test days)."""
    import csv
    path = os.path.join(rd.MODEL_DIR, "test_results.csv")
    if not os.path.exists(path):
        return {}
    return {float(r["cut_off"]): {k: float(v) for k, v in r.items()} for r in csv.DictReader(open(path))}


def rd_standard_cut_off():
    import json
    return json.load(open(os.path.join(rd.MODEL_DIR, "settings.json")))["cut_off"]


def parse_cut_off(text):
    m = re.match(r"\s*([01]?[.,]?\d*)", text or "")
    try:
        return float(m.group(1).replace(",", ".")) if m and m.group(1) not in ("", ".", ",") else None
    except ValueError:
        return None


class CutoffDialog:
    def __init__(self, app):
        self.app = app
        w = self.w = tk.Toplevel(app.root)
        w.title("Change the cut-off")
        w.transient(app.root)
        w.resizable(False, False)
        f = ttk.Frame(w, padding=12)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Rewrite the Raven tables of finished results with another cut-off. It takes seconds and "
                          "doesn't need the recordings.", wraplength=520).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(f, text="Results folder:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        start = app.out.get() if os.path.exists(os.path.join(app.out.get(), "detector.json")) else ""
        self.folder = tk.StringVar(value=start)
        ttk.Entry(f, textvariable=self.folder, width=55).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(f, text="Choose...", command=self.pick).grid(row=2, column=2, padx=(6, 0))
        ttk.Label(f, text="New cut-off (pick one, or type any number from 0 to 1):").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.curve = load_test_results()
        std = rd_standard_cut_off()
        names = [f"{c:g}" + ("  (standard)" if abs(c - std) < 1e-9 else "") for c in sorted(self.curve)]
        self.value = tk.StringVar(value=f"{std:g}  (standard)")
        box = ttk.Combobox(f, textvariable=self.value, values=names, width=18, height=len(names))
        box.grid(row=4, column=0, sticky="w")
        self.value.trace_add("write", lambda *_: self.explain())
        self.details = ttk.Label(f, font=("Consolas", 9), justify="left")
        self.details.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(f, text="Lower finds more rumbles but gives more false alarms to check; higher gives fewer false "
                          "alarms but misses more rumbles. The full table is in the README.",
                  wraplength=520, style="Hint.TLabel").grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.explain()
        fb = ttk.Frame(f)
        fb.grid(row=7, column=0, columnspan=3, sticky="e", pady=(12, 0))
        self.apply_b = ttk.Button(fb, text="Rewrite the tables", command=self.apply)
        self.apply_b.pack(side="left")
        ttk.Button(fb, text="Close", command=w.destroy).pack(side="left", padx=(6, 0))
        w.grab_set()

    def explain(self):
        v = parse_cut_off(self.value.get())
        if v is None or not 0 <= v <= 1:
            self.details.configure(text="Type a number between 0 and 1, such as 0.95.\n\n\n")
            return
        if not self.curve:
            self.details.configure(text="")
            return
        near = min(self.curve, key=lambda c: abs(c - v))
        r = self.curve[near]
        head = (f"With this model on ELP's test days, at {near:g}" +
                ("" if abs(near - v) < 1e-9 else f" (the nearest tested value to {v:g})") + ":")
        self.details.configure(text=(
            f"{head}\n"
            f"  Sample A (54 Nouabale-Ndoki days): finds {r['sampleA_recall'] * 100:.0f}% of rumbles, "
            f"{r['sampleA_precision'] * 100:.0f}% of detections are real\n"
            f"  Sample B (2024 test days):         finds {r['sampleB_recall'] * 100:.0f}% of rumbles, "
            f"{r['sampleB_precision'] * 100:.0f}% of detections are real\n"
            f"  False alarms per hour: {r['sampleA_false_alarms_per_hour']:.2f} (Sample A), "
            f"{r['vehicle_false_alarms_per_hour']:.2f} (days with vehicles)"))

    def pick(self):
        d = filedialog.askdirectory(parent=self.w, title="Choose the results folder (the one with summary.md)")
        if d:
            self.folder.set(os.path.normpath(d))

    def apply(self):
        out = rd.clean_path(self.folder.get())
        if not os.path.exists(os.path.join(out, "detector.json")):
            messagebox.showwarning("Change the cut-off", "That isn't a results folder (it has no detector.json). "
                                   "Choose the folder that has summary.md in it.", parent=self.w)
            return
        try:
            v = parse_cut_off(self.value.get())
            assert v is not None and 0 <= v <= 1
        except (ValueError, AssertionError):
            messagebox.showwarning("Change the cut-off", "Type a number between 0 and 1, such as 0.95.", parent=self.w)
            return
        if self.app.worker:
            messagebox.showwarning("Change the cut-off", "Wait until the detector has finished.", parent=self.w)
            return
        self.apply_b.configure(state="disabled")
        self.app.clear_log()
        self.app.set_status(f"Rewriting the tables with cut-off {v}...")
        self.app.set_running(True)
        self.app.worker = Worker([PYTHON, os.path.join(HERE, "rumble_detector.py"), "table", "--out", out,
                                  "--min-score", str(v)], self.app.lines, restart=False)
        self.app.out.set(out)
        self.app.out_edited = True
        self.w.destroy()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    try:
        App(root, rd.clean_path(argv[0]) if argv else None)
        root.mainloop()
    except Exception:  # noqa: BLE001 - pythonw has no console: say what happened
        err = traceback.format_exc()
        with open(os.path.join(HERE, "gui_error.txt"), "w", encoding="utf-8") as f:
            f.write(err)
        messagebox.showerror(TITLE, f"Something went wrong:\n\n{err[-1500:]}\n\nThis is saved in gui_error.txt; "
                                    "please send it to Amogh.")


if __name__ == "__main__":
    main()
