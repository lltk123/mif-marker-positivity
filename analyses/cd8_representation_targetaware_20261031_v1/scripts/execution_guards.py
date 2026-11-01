"""Atomic artifacts and bounded, persistent execution accounting. No model logic."""
import os, json, time, hashlib, uuid, signal, subprocess
from pathlib import Path
from contextlib import contextmanager
import numpy as np


class BudgetExhausted(RuntimeError):
    pass


class WatchdogExpired(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(str(record))


def _default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("wb") as f:
        np.save(f, np.asarray(array), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def atomic_checkpoint(path, value):
    import torch

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_keys_hash(frame):
    h = hashlib.sha256()
    for dataset, image, cell, marker in frame[
        ["dataset_id", "image_id", "cell_id", "marker"]
    ].itertuples(index=False, name=None):
        parts = [str(dataset), str(image), str(int(cell)), str(marker)]
        if any("\x1f" in x or "\n" in x for x in parts):
            raise ValueError("Unsupported key separator")
        h.update(("\x1f".join(parts) + "\n").encode("utf-8"))
    return h.hexdigest()


def native_threshold(threshold, dtype):
    return float(np.asarray(threshold, dtype=np.dtype(dtype)))


def validate_score(path, meta_path, rows, expected):
    path = Path(path)
    meta_path = Path(meta_path)
    if not path.exists() or not meta_path.exists():
        return False, "missing array or completion metadata"
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.shape != (int(rows),):
            return False, "shape mismatch"
        if not np.isfinite(a).all():
            return False, "nonfinite scores"
        if m.get("rows") != int(rows) or m.get("dtype") != str(a.dtype):
            return False, "row/dtype metadata mismatch"
        for k, v in expected.items():
            if m.get(k) != v:
                return False, "signature mismatch: " + k
        if m.get("score_sha256") != file_sha(path):
            return False, "score hash mismatch"
        return True, "verified shape, finite, key/input/model signatures and SHA256"
    except Exception as e:
        return False, type(e).__name__ + ": " + str(e)


def save_score(path, meta_path, array, signature, extra=None):
    a = np.asarray(array)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("Scores must be one-dimensional and entirely finite")
    atomic_npy(path, a)
    m = dict(extra or {})
    m.update(signature)
    m.update(
        schema_version=2,
        rows=len(a),
        dtype=str(a.dtype),
        score_sha256=file_sha(path),
        complete=True,
        all_finite=True,
    )
    atomic_json(meta_path, m)
    return m


@contextmanager
def score_lock(path):
    import fcntl

    p = Path(str(path) + ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another writer owns " + str(path))
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def quarantine(path, run_root, reason):
    path = Path(path)
    root = Path(run_root).resolve()
    if root not in path.resolve().parents:
        raise ValueError("Refuse to move an artifact outside the named run")
    if not path.exists():
        return None
    target = root / "provenance/incomplete_artifacts"
    target.mkdir(parents=True, exist_ok=True)
    dest = target / (path.name + "." + uuid.uuid4().hex)
    os.replace(str(path), str(dest))
    atomic_json(
        str(dest) + ".json",
        {"original": str(path), "reason": reason, "time": time.time()},
    )
    return dest


def pid_alive(pid):
    pid = int(pid)
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class BudgetLedger:
    def __init__(self, path, cap=28800.0, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.last_persist = 0.0
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"consumed_seconds": 0.0, "hard_cap_seconds": float(cap)}
        )
        self.data.setdefault(
            "status", "partial" if self.data.get("consumed_seconds", 0) else "ready"
        )
        self.data.setdefault("events", [])
        active = self.data.get("active")
        if active and int(active.get("pid", 0)) != os.getpid():
            if pid_alive(active.get("pid", 0)):
                raise RuntimeError("A live process already owns the execution budget")
            charge = max(0.0, self.clock() - float(active["last_tick_wall"]))
            self.data["consumed_seconds"] += charge
            self.data["events"].append(
                {
                    "event": "interrupted_run_recovered",
                    "charged_upper_bound_seconds": charge,
                    "accounting": "includes uncertain downtime conservatively; no lost retry time",
                }
            )
            self.data["active"] = None
            self.data["status"] = "partial"
            self.persist()

    def persist(self):
        atomic_json(self.path, self.data)
        self.last_persist = self.clock()

    def begin(self, name):
        if self.data.get("active"):
            raise RuntimeError("Unclosed budget activity")
        if self.data["consumed_seconds"] >= self.data["hard_cap_seconds"]:
            self.data["status"] = "budget_exhausted"
            self.persist()
            raise BudgetExhausted("No remaining frozen budget")
        now = self.clock()
        self.data["active"] = {
            "pid": os.getpid(),
            "model": name,
            "started_wall": now,
            "last_tick_wall": now,
        }
        self.data["status"] = "running"
        self.persist()

    def checkpoint(self, force=False):
        active = self.data.get("active")
        if not active:
            return
        now = self.clock()
        self.data["consumed_seconds"] += max(0.0, now - float(active["last_tick_wall"]))
        active["last_tick_wall"] = now
        over = max(0.0, self.data["consumed_seconds"] - self.data["hard_cap_seconds"])
        self.data["overrun_seconds"] = over
        if self.data["consumed_seconds"] >= self.data["hard_cap_seconds"]:
            self.data["status"] = "budget_exhausted"
            self.persist()
            raise BudgetExhausted(
                "Frozen wall budget reached; bounded in-flight tail seconds="
                + str(over)
            )
        if force or now - self.last_persist >= 5.0:
            self.persist()

    def finish(self, status="running"):
        self.checkpoint(force=True)
        self.data["active"] = None
        self.data["status"] = status
        self.persist()

    def abort(self, error):
        active = self.data.get("active")
        if active:
            now = self.clock()
            self.data["consumed_seconds"] += max(
                0.0, now - float(active["last_tick_wall"])
            )
        self.data["active"] = None
        self.data["status"] = (
            "budget_exhausted"
            if self.data["consumed_seconds"] >= self.data["hard_cap_seconds"]
            else "partial"
        )
        self.data["events"].append(
            {"event": "activity_failed", "error": str(error), "time": self.clock()}
        )
        self.persist()

    @property
    def remaining(self):
        return max(0.0, self.data["hard_cap_seconds"] - self.data["consumed_seconds"])


def wait_bounded(proc, seconds, grace=5.0, process_group=True):
    started = time.time()
    try:
        return proc.wait(timeout=max(0.001, float(seconds)))
    except subprocess.TimeoutExpired:
        termination = time.time()
        if process_group and os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        forced = False
        try:
            proc.wait(timeout=float(grace))
        except subprocess.TimeoutExpired:
            forced = True
            if process_group and os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5.0)
        raise WatchdogExpired(
            {
                "pid": proc.pid,
                "status": "budget_exhausted_or_timeout",
                "allowed_seconds": float(seconds),
                "wall_seconds": time.time() - started,
                "shutdown_tail_seconds": time.time() - termination,
                "forced_kill": forced,
            }
        )


def neural_completion_state(returncode, complete, budget_flag):
    if returncode == 0 and complete and int(complete.get("n_models", 0)) == 15:
        return "completed"
    if budget_flag:
        return "partial_budget_exhausted"
    if returncode == 0:
        return "partial_missing_completion_marker"
    return "failed"


def signature_for(d, manifest, selection, name):
    from experiment_common import RUN, sha

    frozen = json.loads(Path(selection).read_text())
    models = frozen["model_file_sha256"]
    assert (
        _checked_input(RUN / "data/features.parquet") == manifest["feature_sha256"]
    ), "Feature input changed"
    for fn in (
        [name + ".pt"]
        if name.startswith("N_")
        else [name + ".joblib"]
        if name.startswith("H")
        else [
            x + ".pt"
            for x in json.loads((RUN / "models" / (name + ".json")).read_text())[
                "members"
            ]
        ]
    ):
        assert file_sha(RUN / "models" / fn) == models[fn], (
            "Frozen model changed: " + fn
        )
    member = models.get(name + ".pt", models.get(name + ".joblib"))
    if member is None:
        members = json.loads((RUN / "models" / (name + ".json")).read_text())[
            "member_model_sha256"
        ]
        member = hashlib.sha256(
            json.dumps(members, sort_keys=True).encode()
        ).hexdigest()
    return {
        "keys_sha256": canonical_keys_hash(d),
        "feature_sha256": manifest["feature_sha256"],
        "selection_sha256": file_sha(selection),
        "model_sha256": member,
    }


_INPUT_CACHE = {}


def _checked_input(path):
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _INPUT_CACHE:
        _INPUT_CACHE[key] = file_sha(p)
    return _INPUT_CACHE[key]
