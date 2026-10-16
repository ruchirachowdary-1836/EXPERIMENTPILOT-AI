"""Sandboxed, time-limited experiment execution.

Experiments run on copies of the uploaded data in a worker *process* (default) so that a slow or
crashing experiment cannot take down the API and can be abandoned after a timeout. Workers receive
only file paths and plain configuration, load the CSV copies themselves, never touch the network,
and run single-threaded numerics.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from ..core.config import get_settings


class ExperimentTimeout(RuntimeError):
    pass


def _worker_init() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def execute_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    """Runs inside the worker. ``payload`` holds file paths, mapping and the experiment id."""
    import pandas as pd

    from experimentpilot.experiments import ColumnMapping, ExperimentContext, get_experiment

    ref = pd.read_csv(payload["reference_path"])
    rec = pd.read_csv(payload["recent_path"])
    rep = pd.read_csv(payload["replay_path"]) if payload.get("replay_path") else None
    ctx = ExperimentContext(ref, rec, ColumnMapping(**payload["mapping"]), rep, payload.get("reported_metrics"),
                            payload.get("config", {}), int(payload.get("seed", 0)))
    return get_experiment(payload["experiment_id"]).run(ctx).to_dict()


class ExperimentExecutor:
    def __init__(self) -> None:
        self._pool: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def _get_pool(self):
        with self._lock:
            if self._pool is None:
                if get_settings().executor == "thread":
                    self._pool = ThreadPoolExecutor(max_workers=2)
                else:
                    self._pool = ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"),
                                                     initializer=_worker_init)
            return self._pool

    def run(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        timeout = timeout or get_settings().experiment_timeout_s
        fut: Future = self._get_pool().submit(execute_experiment, payload)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout as exc:
            fut.cancel()
            self.reset()
            raise ExperimentTimeout(f"the experiment exceeded the {timeout:.0f}s time limit") from exc

    def reset(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is None:
            return
        if isinstance(pool, ProcessPoolExecutor):
            for p in list(getattr(pool, "_processes", {}).values()):
                try:
                    p.terminate()
                except Exception:
                    pass
        pool.shutdown(wait=False, cancel_futures=True)

    def shutdown(self) -> None:
        self.reset()


executor = ExperimentExecutor()
