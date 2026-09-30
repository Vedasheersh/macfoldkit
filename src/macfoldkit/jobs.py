"""Durable, local-only folding jobs for one per-user Apple GPU worker."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid

from .config import PACKAGE, runtime_env, runtime_python, supported_platform

LABEL = "io.github.macfoldkit.jobs"
MODELS = ("alphafold2_ptm", "alphafold2_multimer_v3")
FINISHED = ("succeeded", "failed", "cancelled", "interrupted")
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")


def jobs_home() -> Path:
    return Path.home() / "Library/Application Support/macfoldkit"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def weight_path(home: Path, model_type: str) -> Path:
    if model_type not in MODELS:
        raise ValueError(f"Unsupported ColabFold model: {model_type}")
    suffix = "_ptm" if model_type == "alphafold2_ptm" else "_multimer_v3"
    return home / "weights/colabfold/params" / f"params_model_1{suffix}.npz"


def require_assets(home: Path, model_type: str) -> Path:
    python = runtime_python(home, "colabfold")
    if not python.is_file() or not (home / "colabfold.json").is_file():
        raise RuntimeError(f"ColabFold is not set up. Run: macfoldkit --home '{home}' setup colabfold")
    weights = weight_path(home, model_type)
    if not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError(f"Missing local weights: {weights}. Run: macfoldkit --home '{home}' "
                           f"fetch colabfold --model-type {model_type}")
    return python


def validate_input(path: Path, model_type: str) -> str:
    if not path.is_file():
        raise ValueError(f"Input must be a local FASTA or A3M file: {path}")
    extension = path.suffix.lower()
    if extension not in (".fasta", ".fa", ".faa", ".a3m"):
        raise ValueError("Jobs accept one .fasta, .fa, .faa or .a3m file, not directories or other formats")
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except UnicodeError as exc:
        raise ValueError(f"Input is not UTF-8 text: {path}") from exc
    if not lines:
        raise ValueError("Input has no sequences")
    metadata = None
    if extension == ".a3m" and lines[0].startswith("#"):
        metadata = lines.pop(0)
        if not re.fullmatch(r"#[0-9]+(?:,[0-9]+)*\t[0-9]+(?:,[0-9]+)*", metadata):
            raise ValueError("Invalid multimer A3M header")
        lengths, counts = (part.split(",") for part in metadata[1:].split("\t"))
        if len(lengths) != len(counts) or any(int(count) == 0 for count in counts):
            raise ValueError("Invalid multimer A3M chain counts")
        if len(lengths) > 1 and model_type != "alphafold2_multimer_v3":
            raise ValueError("Multimer A3M requires --model-type alphafold2_multimer_v3")
    records = []
    for line in lines:
        if line.startswith(">"):
            if len(line) == 1:
                raise ValueError("Empty FASTA/A3M header")
            records.append("")
        elif not records or line.startswith("#"):
            raise ValueError("Expected a FASTA/A3M header before each sequence")
        else:
            records[-1] += line
    if not records or any(not sequence for sequence in records):
        raise ValueError("Missing FASTA/A3M sequence")
    if extension != ".a3m":
        if metadata or len(records) != 1:
            raise ValueError("A folding job accepts exactly one FASTA record")
        chains = records[0].upper().split(":")
        if any(not re.fullmatch(r"[A-Z]+", chain) for chain in chains):
            raise ValueError("FASTA must contain nonempty protein chains separated by ':'")
        if len(chains) > 1 and model_type != "alphafold2_multimer_v3":
            raise ValueError("Multiple chains require --model-type alphafold2_multimer_v3")
    else:
        if not re.fullmatch(r"[A-Z]+", records[0]):
            raise ValueError("A3M must start with a complete, ungapped uppercase query sequence")
        length = len(records[0])
        if metadata and sum(map(int, metadata[1:].split("\t")[0].split(","))) != length:
            raise ValueError("A3M multimer header does not match query length")
        for sequence in records:
            if not re.fullmatch(r"[A-Za-z.-]+", sequence):
                raise ValueError("Invalid A3M alignment row")
            if len(re.sub(r"[a-z.]", "", sequence)) != length:
                raise ValueError("A3M row does not cover the entire query alignment")
    return extension


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (root / "jobs").mkdir(exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, started_at TEXT, ended_at TEXT,
                source_name TEXT NOT NULL, extension TEXT NOT NULL, sha256 TEXT NOT NULL,
                runtime_home TEXT NOT NULL, model_type TEXT NOT NULL,
                num_recycle INTEGER NOT NULL, seed INTEGER NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
                pid INTEGER, returncode INTEGER, error TEXT
            )""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def folder(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise ValueError("Invalid job ID; expected 32 lowercase hexadecimal characters")
        return self.root / "jobs" / job_id

    def get(self, job_id: str) -> dict:
        self.folder(job_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown job: {job_id}")
        return dict(row)

    def all(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY rowid DESC")]

    def submit(self, source: Path, home: Path, model_type: str, num_recycle: int, seed: int) -> dict:
        if num_recycle < 0 or seed < 0:
            raise ValueError("Recycle count and seed must be nonnegative")
        require_assets(home, model_type)
        extension = validate_input(source, model_type)
        job_id = uuid.uuid4().hex
        folder = self.folder(job_id)
        folder.mkdir(mode=0o700)
        copied = folder / f"input{extension}"
        try:
            shutil.copyfile(source, copied)
            validate_input(copied, model_type)
            with copied.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            timestamp = now()
            with self.connect() as db:
                db.execute("""INSERT INTO jobs
                    (id, status, created_at, updated_at, source_name, extension, sha256,
                     runtime_home, model_type, num_recycle, seed)
                    VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (job_id, timestamp, timestamp, source.name, extension, digest,
                     str(home), model_type, num_recycle, seed))
        except Exception:
            shutil.rmtree(folder)
            raise
        return self.get(job_id)

    def change(self, job_id: str, operation: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise ValueError(f"Unknown job: {job_id}")
            status = job["status"]
            if operation == "cancel":
                if status == "pending":
                    db.execute("UPDATE jobs SET status='cancelled', updated_at=?, ended_at=? WHERE id=?",
                               (now(), now(), job_id))
                elif status == "running":
                    db.execute("UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?", (now(), job_id))
                else:
                    raise ValueError(f"Cannot cancel a {status} job")
            elif operation == "retry":
                if status not in ("failed", "cancelled", "interrupted"):
                    raise ValueError(f"Cannot retry a {status} job")
                db.execute("""UPDATE jobs SET status='pending', updated_at=?, started_at=NULL,
                    ended_at=NULL, pid=NULL, returncode=NULL, error=NULL, cancel_requested=0
                    WHERE id=?""", (now(), job_id))
            else:
                raise ValueError(f"Unknown job operation: {operation}")
        return self.get(job_id)

    def delete(self, job_id: str) -> None:
        self.folder(job_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown job: {job_id}")
            if row["status"] == "running":
                raise ValueError("Cancel and wait for a running job to stop before deleting it")
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        shutil.rmtree(self.folder(job_id))

    def claim(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='pending' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("""UPDATE jobs SET status='running', attempt=attempt+1, started_at=?,
                ended_at=NULL, updated_at=?, cancel_requested=0, pid=NULL WHERE id=?""",
                (now(), now(), row["id"]))
        return self.get(row["id"])

    def recover(self) -> None:
        with self.connect() as db:
            for row in db.execute("SELECT pid FROM jobs WHERE status='running' AND pid IS NOT NULL"):
                try:
                    os.kill(row["pid"], 0)
                except ProcessLookupError:
                    continue
                except PermissionError:
                    pass
                raise RuntimeError(f"Previous folding process {row['pid']} may still be running; "
                                   "inspect it before starting another worker")
            db.execute("""UPDATE jobs SET status='interrupted', ended_at=?, updated_at=?, pid=NULL,
                error='Worker stopped during execution; retry explicitly'
                WHERE status='running'""", (now(), now()))

    def update(self, job_id: str, **fields) -> None:
        allowed = {"status", "pid", "returncode", "error", "ended_at"}
        if not fields.keys() <= allowed:
            raise ValueError("Invalid job update")
        fields["updated_at"] = now()
        names = ", ".join(f"{key}=?" for key in fields)
        with self.connect() as db:
            db.execute(f"UPDATE jobs SET {names} WHERE id=?", (*fields.values(), job_id))

    def storage_bytes(self, job_id: str | None = None) -> int:
        path = self.folder(job_id) if job_id else self.root / "jobs"
        return sum(Path(directory, name).stat().st_size for directory, _, names in os.walk(path)
                   for name in names)

    def details(self, job_id: str) -> dict:
        job = self.get(job_id)
        folder = self.folder(job_id)
        job["input"] = str(folder / f"input{job['extension']}")
        job["results"] = str(folder / "attempts" / str(job["attempt"]) / "output") if job["attempt"] else None
        job["log"] = str(folder / "attempts" / str(job["attempt"]) / "run.log") if job["attempt"] else None
        job["storage_bytes"] = self.storage_bytes(job_id)
        return job


def _terminate_group(process: subprocess.Popen) -> int:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait()


def execute(store: JobStore, job: dict, stop: list[bool]) -> None:
    job_id = job["id"]
    folder = store.folder(job_id)
    attempt = folder / "attempts" / str(job["attempt"])
    attempt.mkdir(parents=True)
    log_path = attempt / "run.log"
    process = None
    with log_path.open("w") as log:
        try:
            home = Path(job["runtime_home"])
            python = require_assets(home, job["model_type"])
            source = folder / f"input{job['extension']}"
            validate_input(source, job["model_type"])
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != job["sha256"]:
                    raise ValueError("Copied input changed since submission")
            mode = "mmseqs2_uniref_env" if job["extension"] == ".a3m" else "single_sequence"
            env = runtime_env(home, "colabfold")
            env["HF_HUB_OFFLINE"] = "1"
            env.pop("HF_TOKEN", None)
            env.pop("HF_ENDPOINT", None)
            if stop[0] or store.get(job_id)["cancel_requested"]:
                status = "interrupted" if stop[0] else "cancelled"
                store.update(job_id, status=status, ended_at=now(), pid=None)
                return
            command = ["/usr/bin/caffeinate", "-i", str(python),
                       str(PACKAGE / "runners/colabfold/run_mps.py"),
                       str(source), str(attempt / "output"), "--local-only",
                       "--msa-mode", mode, "--model-type", job["model_type"],
                       "--num-recycle", str(job["num_recycle"]),
                       "--random-seed", str(job["seed"])]
            print(f"Starting local {job['model_type']} fold (attempt {job['attempt']})", file=log, flush=True)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            store.update(job_id, pid=process.pid)
            while process.poll() is None:
                if stop[0] or store.get(job_id)["cancel_requested"]:
                    _terminate_group(process)
                    status = "interrupted" if stop[0] else "cancelled"
                    error = "Worker stopped; retry explicitly" if stop[0] else "Cancelled by user"
                    print(error, file=log, flush=True)
                    store.update(job_id, status=status, ended_at=now(), error=error, pid=None,
                                 returncode=process.returncode)
                    return
                time.sleep(0.5)
            if store.get(job_id)["cancel_requested"]:
                store.update(job_id, status="cancelled", ended_at=now(), error="Cancelled by user",
                             returncode=process.returncode, pid=None)
            elif process.returncode:
                store.update(job_id, status="failed", ended_at=now(),
                             error=f"ColabFold exited with code {process.returncode}; see {log_path}",
                             returncode=process.returncode, pid=None)
            else:
                store.update(job_id, status="succeeded", ended_at=now(), returncode=0, pid=None)
        except Exception as exc:
            if process is not None and process.poll() is None:
                _terminate_group(process)
            traceback.print_exc(file=log)
            store.update(job_id, status="failed", ended_at=now(), pid=None,
                         error=f"{type(exc).__name__}: {exc}")


def worker(store: JobStore, once: bool = False) -> None:
    if not supported_platform():
        raise RuntimeError("The GPU worker requires Apple Silicon macOS")
    lock_path = store.root / "worker.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A MacFoldKit jobs worker is already running") from exc
        stop = [False]
        previous = {}

        def shutdown(_signum, _frame):
            stop[0] = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, shutdown)
        try:
            store.recover()
            while not stop[0]:
                job = store.claim()
                if job:
                    execute(store, job, stop)
                elif once:
                    break
                else:
                    time.sleep(1)
                if once:
                    break
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _loaded() -> bool:
    return subprocess.run(["/bin/launchctl", "print", f"gui/{os.getuid()}/{LABEL}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def install_worker(store: JobStore) -> dict:
    if not supported_platform():
        raise RuntimeError("The login worker requires Apple Silicon macOS")
    agents = Path.home() / "Library/LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    path = agents / f"{LABEL}.plist"
    config = {"Label": LABEL, "ProgramArguments": [sys.executable, "-m", "macfoldkit.cli",
              "jobs", "worker"], "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 5,
              "StandardOutPath": str(store.root / "worker.log"),
              "StandardErrorPath": str(store.root / "worker.log"),
              "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONUNBUFFERED": "1"}}
    if path.exists():
        if plistlib.loads(path.read_bytes()) != config:
            raise RuntimeError(f"{path} uses another Python or configuration. Stop active jobs, "
                               "run 'macfoldkit jobs uninstall', then install again.")
    else:
        path.write_bytes(plistlib.dumps(config))
    if not _loaded():
        subprocess.run(["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    return {"installed": True, "launch_agent": str(path), "python": sys.executable, "loaded": _loaded()}


def uninstall_worker() -> dict:
    path = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    if _loaded():
        subprocess.run(["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], check=True)
    path.unlink(missing_ok=True)
    return {"installed": False, "jobs_retained": True}


def run_command(args, home: Path) -> None:
    store = JobStore(jobs_home())
    action = args.jobs_command
    if action == "submit":
        result = store.details(store.submit(Path(args.input).expanduser(), home,
                                             args.model_type, args.num_recycle, args.seed)["id"])
    elif action == "list":
        result = {"jobs": [store.details(row["id"]) for row in store.all()],
                  "storage_bytes": store.storage_bytes()}
    elif action == "status":
        result = store.details(args.id) if args.id else {
            "jobs": {status: sum(row["status"] == status for row in store.all())
                     for status in ("pending", "running", *FINISHED)},
            "storage_bytes": store.storage_bytes(), "home": str(store.root),
            "worker_loaded": _loaded() if supported_platform() else False}
    elif action == "logs":
        if args.tail < 0:
            raise ValueError("--tail must be nonnegative")
        job = store.details(args.id)
        path = Path(job["log"]) if job["log"] else None
        if path is None or not path.is_file():
            raise ValueError("This job has no run log yet")
        with path.open() as stream:
            text = stream.read() if args.all else "".join(deque(stream, maxlen=args.tail))
        if args.json:
            result = {"id": args.id, "log": text}
        else:
            print(text, end="")
            return
    elif action in ("cancel", "retry"):
        result = store.details(store.change(args.id, action)["id"])
    elif action == "delete":
        if not args.yes:
            raise ValueError("Deletion permanently removes inputs, logs and results; pass --yes")
        store.delete(args.id)
        result = {"deleted": args.id}
    elif action == "install":
        result = install_worker(store)
    elif action == "uninstall":
        result = uninstall_worker()
    elif action == "worker":
        worker(store, once=args.once)
        return
    else:
        raise ValueError(f"Unknown jobs command: {action}")
    print(json.dumps(result, indent=2))


def add_arguments(parent) -> None:
    actions = parent.add_subparsers(dest="jobs_command", required=True)
    submit = actions.add_parser("submit", help="Queue one local ColabFold fold")
    submit.add_argument("input", help="One local FASTA or complete A3M file")
    submit.add_argument("--model-type", choices=MODELS, default="alphafold2_ptm")
    submit.add_argument("--num-recycle", type=int, default=3)
    submit.add_argument("--seed", type=int, default=7)
    actions.add_parser("list", help="List jobs and retained disk use")
    status = actions.add_parser("status", help="Show worker/queue status or one job")
    status.add_argument("id", nargs="?")
    logs = actions.add_parser("logs", help="Show the latest attempt's log")
    logs.add_argument("id")
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--all", action="store_true")
    logs.add_argument("--json", action="store_true")
    for name in ("cancel", "retry", "delete"):
        action = actions.add_parser(name)
        action.add_argument("id")
        if name == "delete":
            action.add_argument("--yes", action="store_true")
    actions.add_parser("install", help="Start the worker at login with launchd")
    actions.add_parser("uninstall", help="Stop and remove the worker; preserve jobs")
    service = actions.add_parser("worker", help="Run worker in foreground (launchd uses this)")
    service.add_argument("--once", action="store_true", help="Process at most one queued job")
