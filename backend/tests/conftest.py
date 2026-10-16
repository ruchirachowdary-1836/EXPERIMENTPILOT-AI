import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import reset_settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EP_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("EP_EXECUTOR", "thread")
    monkeypatch.setenv("EP_MAX_UPLOAD_MB", "1")
    reset_settings()
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c
    reset_settings()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def make_frames(n=400, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = (x1 + 0.5 * x2 + rng.normal(0, 0.7, n) > 0).astype(int)
    df = pd.DataFrame({"x1": x1 + shift, "x2": x2, "cat": rng.choice(["a", "b", "c"], n), "target": y})
    df["prediction"] = ((x1 + 0.5 * x2) > 0).astype(int)
    return df


@pytest.fixture()
def demo_project(client):
    """A project created from a simulated pipeline-change scenario."""
    r = client.post("/api/v1/demo", json={"dataset": "synthetic_credit", "seed": 3,
                                          "faults": [{"fault": "pipeline_change", "severity": "high"}]})
    assert r.status_code == 201, r.text
    return r.json()
