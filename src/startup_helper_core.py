"""
Environment pre-checks, startup validation, and error-text diagnosis.
(Does NOT import app/main to avoid pulling heavy dependencies before checks.)
"""
from __future__ import annotations

import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


def src_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return src_dir().parent


def requirements_path() -> Path:
    return src_dir() / "requirements.txt"


PIP_TO_IMPORT: dict[str, str] = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "jinja2": "jinja2",
    "playwright": "playwright",
    "requests": "requests",
    "beautifulsoup4": "bs4",
    "python-dotenv": "dotenv",
    "streamlit": "streamlit",
    "bcrypt": "bcrypt",
    "itsdangerous": "itsdangerous",
}


def _parse_requirement_names(req_file: Path) -> list[str]:
    if not req_file.is_file():
        return []
    names: list[str] = []
    for line in req_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_.-]+)", line)
        if m:
            names.append(m.group(1).lower().replace("_", "-"))
    return names


def _import_name_for_pip(pkg: str) -> str:
    return PIP_TO_IMPORT.get(pkg.lower(), pkg.lower().replace("-", "_"))


@dataclass
class CheckItem:
    id: str
    ok: bool
    title: str
    detail: str
    severity: str
    fix_commands: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    code: str
    title: str
    summary: str
    hints: list[str]
    commands: list[str]


def diagnose_error_text(text: str) -> Diagnosis | None:
    if not (text or "").strip():
        return None
    t = text

    def dm(
        code: str,
        title: str,
        summary: str,
        hints: list[str],
        commands: list[str],
    ) -> Diagnosis:
        return Diagnosis(code, title, summary, hints, commands)

    m = re.search(r"No module named\s+['\"]([^'\"]+)['\"]", t, re.I)
    if m:
        mod = m.group(1)
        return dm(
            "MISSING_MODULE",
            "Missing Python Package",
            f"The current environment is missing module: {mod}.",
            [
                'Use the "Install pip Dependencies" button to install requirements.txt.',
                "Make sure the Python used here matches the one used to run the main app (recommended: always use run_helper.bat).",
            ],
            [f'{sys.executable} -m pip install -r "{requirements_path()}"'],
        )

    rules: list[tuple[re.Pattern[str], Diagnosis]] = [
        (
            re.compile(
                r"Python not found|不是内部或外部命令|\'python\' 不是|python不是内部或外部命令",
                re.I,
            ),
            dm(
                "NO_PYTHON",
                "Python Not Found",
                "No usable 'python' command found on this system, or Python is not installed.",
                [
                    "Install Python 3.10+ from python.org and check 'Add Python to PATH'.",
                    "After installation, reopen this helper.",
                ],
                [],
            ),
        ),
        (
            re.compile(r"ModuleNotFoundError", re.I),
            dm(
                "MISSING_MODULE_GENERIC",
                "Import Failed",
                "A Python module could not be found. Dependencies are likely incomplete.",
                [
                    'Click "Install pip Dependencies".',
                    "If using a virtual environment, activate it before launching this helper.",
                ],
                [],
            ),
        ),
        (
            re.compile(
                r"playwright install|Executable doesn\'t exist|BrowserType\.launch|install msedge",
                re.I,
            ),
            dm(
                "PLAYWRIGHT_EDGE",
                "Playwright / Edge",
                "Playwright browser components are not ready, or the Edge channel cannot be launched.",
                [
                    'Click "Install Playwright Browser".',
                    "Make sure Microsoft Edge is installed on this machine.",
                ],
                [f'"{sys.executable}" -m playwright install msedge'],
            ),
        ),
        (
            re.compile(
                r"Address already in use|10048|Only one usage of each socket address",
                re.I,
            ),
            dm(
                "PORT_IN_USE",
                "Port 8001 In Use",
                "Port 8001 is already occupied; cannot start another web service.",
                ["Close the previous command-line window running the main app, or kill the process using port 8001."],
                [],
            ),
        ),
        (
            re.compile(
                r"sqlite3\.OperationalError|database is locked|unable to open database file",
                re.I,
            ),
            dm(
                "SQLITE",
                "Database File Error",
                "Cannot write to or lock the SQLite database (user.db).",
                ["Make sure the src directory is writable.", "Close any other running instances of this program."],
                [],
            ),
        ),
        (
            re.compile(
                r"cannot locate search box|Cookie expired|cookie_expired|debug_searchbox_final\.png",
                re.I,
            ),
            dm(
                "OWA_COOKIE",
                "Mailbox / Cookie Issue",
                "The cookie has expired or the page structure has changed.",
                ['Re-paste the Cookie JSON in the web UI "Settings" page.', "Refer to the maintenance guide."],
                [],
            ),
        ),
        (
            re.compile(r"missing base_url or api_key|缺少 base_url 或 api_key", re.I),
            dm(
                "LLM_CONFIG",
                "LLM Not Configured",
                "The API base_url or api_key has not been configured.",
                ['Fill in the key and endpoint URL in the web UI "Settings" page.'],
                [],
            ),
        ),
        (
            re.compile(r"Failed to launch browser after", re.I),
            dm(
                "BROWSER_LAUNCH",
                "Browser Launch Failed",
                "Playwright was unable to launch the browser.",
                [
                    "Install the Playwright msedge component first.",
                    "Check whether Edge can be opened manually.",
                ],
                [f'"{sys.executable}" -m playwright install msedge'],
            ),
        ),
    ]
    for cre, diag in rules:
        if cre.search(t):
            return diag
    return dm(
        "UNKNOWN",
        "Could Not Identify Error",
        "No built-in rule matched. Send the full error to the maintainer or consult docs/README.md.",
        [
            'Try in order: Install pip Dependencies → Install Playwright Browser → Re-check.',
            "Make sure you launched via run_helper.bat and the project folder structure is intact.",
        ],
        [],
    )


def check_project_layout() -> CheckItem:
    root = repo_root()
    src = src_dir()
    need = [src / "app.py", src / "main.py", requirements_path()]
    missing = [p.relative_to(root) for p in need if not p.is_file()]
    if missing:
        return CheckItem(
            id="layout",
            ok=False,
            title="Project Files Incomplete",
            detail="Missing: " + ", ".join(str(m) for m in missing),
            severity="error",
            fix_commands=[],
        )
    return CheckItem(
        id="layout",
        ok=True,
        title="Project Structure",
        detail=f"Root directory: {root}",
        severity="ok",
        fix_commands=[],
    )


def check_python_version() -> CheckItem:
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        return CheckItem(
            id="python_ver",
            ok=False,
            title="Python Version",
            detail=f"Current: {v.major}.{v.minor}. Recommended: 3.10+.",
            severity="warn",
            fix_commands=[],
        )
    return CheckItem(
        id="python_ver",
        ok=True,
        title="Python Version",
        detail=f"{v.major}.{v.minor}.{v.micro} — {sys.executable}",
        severity="ok",
        fix_commands=[],
    )


def check_venv() -> CheckItem:
    root = repo_root()
    p_venv = root / ".venv" / "Scripts" / "python.exe"
    p_alt = root / "venv" / "Scripts" / "python.exe"
    if p_venv.is_file():
        same = p_venv.resolve() == Path(sys.executable).resolve()
        return CheckItem(
            id="venv",
            ok=True,
            title="Virtual Environment",
            detail=".venv exists"
            + ("; current interpreter is using it." if same else "; not currently active — recommend launching via run_helper.bat."),
            severity="ok" if same else "warn",
            fix_commands=[],
        )
    if p_alt.is_file():
        same = p_alt.resolve() == Path(sys.executable).resolve()
        return CheckItem(
            id="venv",
            ok=True,
            title="Virtual Environment",
            detail="venv exists"
            + ("; current interpreter is using it." if same else "; recommend launching via run_helper.bat."),
            severity="ok" if same else "warn",
            fix_commands=[],
        )
    return CheckItem(
        id="venv",
        ok=True,
        title="Virtual Environment",
        detail="No .venv or venv detected; using system Python.",
        severity="warn",
        fix_commands=[f'"{sys.executable}" -m venv .venv'],
    )


def check_src_writable() -> CheckItem:
    p = src_dir() / ".write_test_startup_helper"
    try:
        p.write_text("ok", encoding="utf-8")
        p.unlink(missing_ok=True)
        return CheckItem(
            id="writable",
            ok=True,
            title="src Directory Writable",
            detail="Can create user.db.",
            severity="ok",
            fix_commands=[],
        )
    except OSError as e:
        return CheckItem(
            id="writable",
            ok=False,
            title="src Directory Not Writable",
            detail=str(e),
            severity="error",
            fix_commands=[],
        )


def check_pip_imports() -> CheckItem:
    req = requirements_path()
    if not req.is_file():
        return CheckItem(
            id="pip_imports",
            ok=False,
            title="Dependency Check",
            detail="Cannot find requirements.txt",
            severity="error",
            fix_commands=[],
        )
    pkgs = _parse_requirement_names(req)
    missing: list[str] = []
    for pkg in pkgs:
        im = _import_name_for_pip(pkg)
        try:
            __import__(im)
        except ImportError:
            missing.append(f"{pkg} (import {im})")
    if missing:
        return CheckItem(
            id="pip_imports",
            ok=False,
            title="Python Dependencies",
            detail="Not installed: " + "; ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""),
            severity="error",
            fix_commands=[
                f'"{sys.executable}" -m pip install -U pip',
                f'"{sys.executable}" -m pip install -r "{requirements_path()}"',
            ],
        )
    return CheckItem(
        id="pip_imports",
        ok=True,
        title="Python Dependencies",
        detail="All packages in requirements.txt can be imported.",
        severity="ok",
        fix_commands=[],
    )


def check_playwright_edge() -> CheckItem:
    """
    Probe Playwright in a separate subprocess. Avoids calling sync_playwright
    directly inside the Tk background thread (not thread-safe; may crash the process).
    """
    script = (
        "import sys\n"
        "try:\n"
        "    from playwright.sync_api import sync_playwright\n"
        "except ImportError as e:\n"
        "    print('ImportError:', e, file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "try:\n"
        "    with sync_playwright() as p:\n"
        "        b = p.chromium.launch(channel='msedge', headless=True)\n"
        "        b.close()\n"
        "except Exception as e:\n"
        "    print(str(e), file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_root()),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return CheckItem(
            id="playwright",
            ok=False,
            title="Playwright / Edge",
            detail="Detection timed out (>30s). Check your network or try again later.",
            severity="error",
            fix_commands=[f'"{sys.executable}" -m playwright install msedge'],
        )
    except OSError as e:
        return CheckItem(
            id="playwright",
            ok=False,
            title="Playwright / Edge",
            detail=f"Cannot start subprocess for detection: {e}",
            severity="error",
            fix_commands=[],
        )
    if r.returncode == 0:
        return CheckItem(
            id="playwright",
            ok=True,
            title="Playwright / Edge",
            detail="msedge channel headless launch succeeded (subprocess check).",
            severity="ok",
            fix_commands=[],
        )
    err = (r.stderr or r.stdout or "").strip()[:500] or f"exit {r.returncode}"
    if r.returncode == 2:
        return CheckItem(
            id="playwright",
            ok=False,
            title="Playwright / Edge",
            detail=f"playwright package not installed: {err}",
            severity="error",
            fix_commands=[],
        )
    return CheckItem(
        id="playwright",
        ok=False,
        title="Playwright / Edge",
        detail=err,
        severity="error",
        fix_commands=[f'"{sys.executable}" -m playwright install msedge'],
    )


def check_port_8001() -> CheckItem:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.4)
        r = sock.connect_ex(("127.0.0.1", 8001))
        if r == 0:
            return CheckItem(
                id="port",
                ok=True,
                title="Port 8001",
                detail="Something is already listening on 8001 (if it's the previous main app, close it first).",
                severity="warn",
                fix_commands=[],
            )
        return CheckItem(
            id="port",
            ok=True,
            title="Port 8001",
            detail="Currently free.",
            severity="ok",
            fix_commands=[],
        )
    finally:
        sock.close()


def iter_all_checks() -> Iterator[CheckItem]:
    """Yield each CheckItem as it completes so the GUI can display incrementally."""
    yield check_project_layout()
    yield check_python_version()
    yield check_venv()
    yield check_src_writable()
    pip_item = check_pip_imports()
    yield pip_item
    if pip_item.ok:
        yield check_playwright_edge()
    else:
        yield CheckItem(
            id="playwright",
            ok=False,
            title="Playwright / Edge",
            detail="Skipped (install pip dependencies first).",
            severity="warn",
            fix_commands=[f'"{sys.executable}" -m playwright install msedge'],
        )
    yield check_port_8001()


def run_all_checks() -> list[CheckItem]:
    """Convenience wrapper that collects all check results into a list."""
    return list(iter_all_checks())


def suggested_pip_install_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", "-r", str(requirements_path())]


def suggested_playwright_install_command() -> list[str]:
    return [sys.executable, "-m", "playwright", "install", "msedge"]


def suggested_venv_create_command() -> list[str]:
    return [sys.executable, "-m", "venv", str(repo_root() / ".venv")]


def iter_subprocess_lines(
    argv: list[str],
    *,
    cwd: Path | None = None,
) -> Iterator[str]:
    cwd = cwd or repo_root()
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout
    for line in proc.stdout:
        yield line.rstrip("\n\r")
    proc.wait()
    yield f"[exit code {proc.returncode}]"
