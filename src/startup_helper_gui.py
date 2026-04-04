"""Graphical setup helper: environment checks, semi-automatic installation, error-paste diagnosis."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from startup_helper_core import (
    diagnose_error_text,
    iter_subprocess_lines,
    repo_root,
    iter_all_checks,
    suggested_pip_install_command,
    suggested_playwright_install_command,
    suggested_venv_create_command,
)

ROOT = repo_root()


class HelperApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Email Assistant - Setup & Diagnostics")
        self.geometry("900x640")
        self.minsize(760, 520)

        self._log_q: queue.Queue[str] = queue.Queue()

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        tab_env = ttk.Frame(nb)
        tab_err = ttk.Frame(nb)
        nb.add(tab_env, text="Environment Check")
        nb.add(tab_err, text="Paste Error Analysis")

        self._build_env_tab(tab_env)
        self._build_err_tab(tab_err)

        self.after(120, self._poll_log)
        self._run_checks_async()

    def _build_env_tab(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(bar, text="Re-check", command=self._run_checks_async).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Create .venv", command=self._on_create_venv).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Install pip Dependencies", command=self._on_pip_install).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Install Playwright Browser", command=self._on_playwright_install).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bar, text="Launch Main App", command=self._on_launch_main).pack(side=tk.LEFT, padx=12)
        ttk.Button(bar, text="Open README", command=self._open_readme).pack(side=tk.RIGHT, padx=2)

        tree_fr = ttk.Frame(parent)
        tree_fr.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            tree_fr,
            columns=("status", "detail"),
            show="tree headings",
            height=9,
        )
        self.tree.heading("#0", text="Check Item")
        self.tree.heading("status", text="Status")
        self.tree.heading("detail", text="Details")
        self.tree.column("#0", width=200)
        self.tree.column("status", width=72)
        self.tree.column("detail", width=580)
        sb = ttk.Scrollbar(tree_fr, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.tag_configure("ok", foreground="#0a6b0a")
        self.tree.tag_configure("warn", foreground="#8a6d1a")
        self.tree.tag_configure("err", foreground="#a01010")

        ttk.Label(parent, text="Install / Command Output (scroll to view)").pack(anchor=tk.W, pady=(8, 0))
        self.log = scrolledtext.ScrolledText(parent, height=11, state=tk.DISABLED, font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True, pady=4)

    def _build_err_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text='Paste the full error text from the console or web page below, then click "Analyze".').pack(anchor=tk.W)
        self.err_in = scrolledtext.ScrolledText(parent, height=14, font=("Consolas", 9))
        self.err_in.pack(fill=tk.BOTH, expand=True, pady=6)
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Analyze", command=self._on_analyze).pack(side=tk.LEFT)
        ttk.Button(bar, text="Clear", command=lambda: self.err_in.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=6)

        ttk.Label(parent, text="Diagnosis & Suggestions").pack(anchor=tk.W, pady=(10, 0))
        self.err_out = scrolledtext.ScrolledText(
            parent, height=12, state=tk.DISABLED, font=("Consolas", 10)
        )
        self.err_out.pack(fill=tk.BOTH, expand=True, pady=4)

    def _append_log(self, line: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _poll_log(self) -> None:
        try:
            while True:
                line = self._log_q.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.after(120, self._poll_log)

    def _run_cmd_logged(self, argv: list[str], cwd: Path | None = None) -> None:
        def worker() -> None:
            try:
                for line in iter_subprocess_lines(argv, cwd=cwd):
                    self._log_q.put(line)
            except Exception as e:
                self._log_q.put(f"[Exception] {e}")
            self._log_q.put('--- Done. Click "Re-check" to verify. ---')

        threading.Thread(target=worker, daemon=True).start()

    def _run_checks_async(self) -> None:
        # Clear the old results immediately
        for ch in self.tree.get_children():
            self.tree.delete(ch)
            
        def worker() -> None:
            try:
                for item in iter_all_checks():
                    self.after(0, lambda it=item: self._append_check_item(it))
            except Exception as e:
                err = str(e)
                self.after(0, lambda msg=err: messagebox.showerror("Check Failed", msg))

        threading.Thread(target=worker, daemon=True).start()

    def _append_check_item(self, it) -> None:
        if not it.ok:
            st = "FAIL"
            tag = "err"
        elif it.severity == "warn":
            st = "WARN"
            tag = "warn"
        else:
            st = "PASS"
            tag = "ok"
        self.tree.insert("", tk.END, text=it.title, values=(st, it.detail), tags=(tag,))

    def _on_create_venv(self) -> None:
        cmd = suggested_venv_create_command()
        if not messagebox.askyesno(
            "Confirm",
            "This will create a .venv in the project root:\n\n" + subprocess.list2cmdline(cmd) + "\n\nProceed?",
        ):
            return
        self._append_log(subprocess.list2cmdline(cmd))
        self._run_cmd_logged(cmd, cwd=ROOT)

    def _on_pip_install(self) -> None:
        cmd = suggested_pip_install_command()
        if not messagebox.askyesno(
            "Confirm",
            "This will run pip install from requirements.txt:\n\n" + subprocess.list2cmdline(cmd) + "\n\nProceed?",
        ):
            return
        self._append_log(subprocess.list2cmdline(cmd))
        self._run_cmd_logged(cmd, cwd=ROOT)

    def _on_playwright_install(self) -> None:
        cmd = suggested_playwright_install_command()
        if not messagebox.askyesno(
            "Confirm",
            "This will download the Playwright Edge component (may take a while):\n\n"
            + subprocess.list2cmdline(cmd)
            + "\n\nProceed?",
        ):
            return
        self._append_log(subprocess.list2cmdline(cmd))
        self._run_cmd_logged(cmd, cwd=ROOT)

    def _on_launch_main(self) -> None:
        bat = ROOT / "run_app.bat"
        if not bat.is_file():
            messagebox.showerror("Not Found", str(bat))
            return
        if not messagebox.askokcancel(
            "Launch Main App",
            "This will open a new command-line window to run run_app.bat (web service).\nYou can keep or close this window.",
        ):
            return
        os.startfile(str(bat))  # type: ignore[attr-defined]

    def _open_readme(self) -> None:
        p = ROOT / "docs" / "README.md"
        if p.is_file():
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Info", f"File not found: {p}")

    def _on_analyze(self) -> None:
        text = self.err_in.get("1.0", tk.END)
        diag = diagnose_error_text(text)
        self.err_out.configure(state=tk.NORMAL)
        self.err_out.delete("1.0", tk.END)
        if not diag:
            self.err_out.insert(tk.END, "Please paste the error text first.")
            self.err_out.configure(state=tk.DISABLED)
            return
        lines = [
            f"Type: {diag.title} ({diag.code})",
            "",
            diag.summary,
            "",
            "Suggestions:",
        ]
        for h in diag.hints:
            lines.append(f"  - {h}")
        if diag.commands:
            lines.append("")
            lines.append("Commands to copy:")
            for c in diag.commands:
                lines.append(c)
        self.err_out.insert(tk.END, "\n".join(lines))
        self.err_out.configure(state=tk.DISABLED)


def _crash_log_path() -> str:
    try:
        return str(repo_root() / "helper_last_error.log")
    except Exception:
        return str(os.path.expanduser("~/practice2_helper_error.log"))


def main() -> None:
    try:
        app = HelperApp()
    except tk.TclError as e:
        print("Cannot start graphical interface (tkinter unavailable):", e, file=sys.stderr)
        print("Please install Python from python.org with Tcl/Tk included, or install dependencies manually.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        log = _crash_log_path()
        try:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            log = "(unable to write log file)"
        try:
            messagebox.showerror(
                "Helper Startup Failed",
                f"{e}\n\nDetails saved to:\n{log}",
            )
        except Exception:
            traceback.print_exc()
        sys.exit(1)
    try:
        app.mainloop()
    except Exception:
        try:
            Path(_crash_log_path()).write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
