import io
import threading

import pandas as pd
import pytest

from tests.conftest import csv_bytes, make_frames

API = "/api/v1"


def upload(client, pid, role, df=None, name="data.csv", raw=None, content_type="text/csv"):
    data = raw if raw is not None else csv_bytes(df)
    return client.post(f"{API}/projects/{pid}/datasets", data={"role": role}, files={"file": (name, data, content_type)})


def new_project(client, name="P1"):
    r = client.post(f"{API}/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()


def start_session(client, pid, **kw):
    body = {"project_id": pid, "name": "S", "budget": 12, **kw}
    r = client.post(f"{API}/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------- basics
def test_health_and_meta(client):
    assert client.get(f"{API}/health").json()["status"] == "ok"
    assert len(client.get(f"{API}/meta/hypotheses").json()) == 8
    exps = client.get(f"{API}/meta/experiments").json()
    assert len(exps) == 10 and all(e["has_likelihood"] for e in exps)
    assert len(client.get(f"{API}/meta/planners").json()) == 7
    assert "provenance" in client.get(f"{API}/meta/likelihoods").json()
    assert client.get(f"{API}/openapi.json").status_code in (200, 404)


def test_project_crud_and_pagination(client):
    for i in range(3):
        new_project(client, f"proj {i}")
    page = client.get(f"{API}/projects?limit=2").json()
    assert page["total"] == 3 and len(page["items"]) == 2
    pid = page["items"][0]["id"]
    assert client.patch(f"{API}/projects/{pid}", json={"name": "renamed"}).json()["name"] == "renamed"
    assert client.get(f"{API}/projects?q=renamed").json()["total"] == 1
    assert client.delete(f"{API}/projects/{pid}").status_code == 204
    assert client.get(f"{API}/projects/{pid}").status_code == 404


def test_validation_errors_have_clear_shape(client):
    r = client.post(f"{API}/projects", json={"name": "   "})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = client.post(f"{API}/sessions", json={"project_id": "x", "budget": -1})
    assert r.status_code == 422


# ---------------------------------------------------------------- upload security
def test_upload_and_schema_mapping(client):
    p = new_project(client)
    df = make_frames()
    r = upload(client, p["id"], "reference", df)
    assert r.status_code == 201
    ds = r.json()["dataset"]
    assert ds["n_rows"] == 400 and {c["name"] for c in ds["schema"]} >= {"x1", "cat", "target"}
    upload(client, p["id"], "recent", make_frames(seed=1, shift=1.0))
    r = client.put(f"{API}/projects/{p['id']}/mapping", json={"target": "target", "prediction": "prediction"})
    assert r.status_code == 200 and r.json()["project"]["ready"] is True
    bad = client.put(f"{API}/projects/{p['id']}/mapping", json={"target": "nope"})
    assert bad.status_code == 422
    prev = client.get(f"{API}/projects/{p['id']}/datasets/reference?rows=5").json()
    assert len(prev["preview"]) == 5


@pytest.mark.parametrize("name,raw,status", [
    ("data.exe", b"a,b\n1,2\n", 415),
    ("data.csv", b"PK\x03\x04binary", 415),
    ("data.csv", b"a,b\n1,\x00\n", 415),
    ("data.csv", b"", 422),
    ("data.csv", b"a\n1\n2\n", 422),                    # one column
    ("data.csv", b"a,a\n1,2\n", 422),                   # duplicate headers
    ("data.csv", "a,b\n1,é\n".encode("latin-1"), 422),  # not utf-8
    ("data.csv", b"a,b\n1,2,3,4\n5,6\n", 422),          # ragged rows
])
def test_upload_rejections(client, name, raw, status):
    p = new_project(client)
    r = upload(client, p["id"], "reference", raw=raw, name=name)
    assert r.status_code == status, r.text
    assert "error" in r.json()


def test_upload_size_limit(client):
    p = new_project(client)
    big = b"a,b\n" + b"1,2\n" * (700_000)       # > 1 MB limit set in the fixture
    r = upload(client, p["id"], "reference", raw=big)
    assert r.status_code == 413


def test_upload_filename_traversal_is_neutralised(client, tmp_path):
    p = new_project(client)
    r = upload(client, p["id"], "reference", make_frames(), name="../../etc/passwd.csv")
    assert r.status_code == 201
    assert r.json()["dataset"]["filename"] == "passwd.csv"
    assert not (tmp_path / "etc").exists()


def test_invalid_role_rejected(client):
    p = new_project(client)
    assert upload(client, p["id"], "evil", make_frames()).status_code == 422


# ---------------------------------------------------------------- sessions
def test_session_requires_data(client):
    p = new_project(client)
    r = client.post(f"{API}/sessions", json={"project_id": p["id"]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_data"


def test_full_diagnosis_flow_updates_posterior_and_reports(client, demo_project):
    pid = demo_project["id"]
    s = start_session(client, pid)
    sid = s["id"]
    assert s["status"] == "active" and s["recommendation"]["selected"]
    prior = {m["hypothesis"]: m["probability"] for m in s["marginals"]}
    seen = []
    for _ in range(3):
        r = client.post(f"{API}/sessions/{sid}/run", json={})
        assert r.status_code == 200, r.text
        seen.append(r.json()["run"]["experiment_id"])
        if r.json()["session"]["status"] != "active":
            break
    assert len(set(seen)) == len(seen)
    sess = client.get(f"{API}/sessions/{sid}").json()
    post = {m["hypothesis"]: m["probability"] for m in sess["marginals"]}
    assert post != prior and sess["spent"] > 0
    hist = client.get(f"{API}/sessions/{sid}/posterior-history").json()["history"]
    assert len(hist) == len(seen) + 1
    ev = client.get(f"{API}/sessions/{sid}/evidence").json()
    assert any(r["tables"] for r in ev["runs"] if r["status"] == "completed")
    tree = client.get(f"{API}/sessions/{sid}/tree").json()
    types = {n["type"] for n in tree["nodes"]}
    assert {"root", "experiment", "outcome"} <= types and any(e["taken"] for e in tree["edges"])
    assert any(not n["taken"] for n in tree["nodes"] if n["type"] == "outcome") or len(seen) == 0
    client.post(f"{API}/sessions/{sid}/stop", json={"reason": "enough"})
    rep = client.get(f"{API}/sessions/{sid}/report").json()
    assert rep["ranking"][0]["posterior"] >= rep["ranking"][-1]["posterior"]
    assert abs(sum(r["posterior"] for r in rep["ranking"]) - sum(m["probability"] for m in sess["marginals"])) < 1e-6
    assert rep["cost"]["spent"] == pytest.approx(sess["spent"]) and rep["limitations"]
    pdf = client.get(f"{API}/sessions/{sid}/export/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF") and len(pdf.content) > 3000
    assert client.get(f"{API}/sessions/{sid}/export/report.json").json()["session"]["id"] == sid
    for kind in ("hypotheses", "experiments", "history"):
        c = client.get(f"{API}/sessions/{sid}/export/{kind}.csv")
        assert c.status_code == 200 and "," in c.text
    assert client.get(f"{API}/sessions/{sid}/export/nope.csv").status_code == 404


def test_pipeline_change_is_found_by_adaptive_planner(client, demo_project):
    s = start_session(client, demo_project["id"], budget=10, confidence_threshold=0.5)
    sid = s["id"]
    for _ in range(8):
        r = client.post(f"{API}/sessions/{sid}/run", json={"approve": True})
        if r.status_code != 200 or r.json()["session"]["status"] != "active":
            break
    rep = client.get(f"{API}/sessions/{sid}/report").json()
    top3 = [r["hypothesis"] for r in rep["ranking"][:3]]
    assert "pipeline_change" in top3
    assert rep["demo_truth"] == ["pipeline_change"]


def test_state_machine(client, demo_project):
    sid = start_session(client, demo_project["id"])["id"]
    assert client.post(f"{API}/sessions/{sid}/pause").json()["status"] == "paused"
    r = client.post(f"{API}/sessions/{sid}/run", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "session_not_active"
    assert client.post(f"{API}/sessions/{sid}/resume").json()["status"] == "active"
    assert client.post(f"{API}/sessions/{sid}/stop", json={"reason": "done"}).json()["status"] == "stopped"
    assert client.post(f"{API}/sessions/{sid}/resume").status_code == 409
    assert client.post(f"{API}/sessions/{sid}/run", json={}).status_code == 409


def test_duplicate_experiment_is_rejected(client, demo_project):
    sid = start_session(client, demo_project["id"])["id"]
    assert client.post(f"{API}/sessions/{sid}/run", json={"experiment_id": "class_prior"}).status_code == 200
    r = client.post(f"{API}/sessions/{sid}/run", json={"experiment_id": "class_prior"})
    assert r.status_code == 409


def test_concurrent_requests_cannot_run_same_experiment_twice(client, demo_project):
    sid = start_session(client, demo_project["id"])["id"]
    codes = []

    def go():
        codes.append(client.post(f"{API}/sessions/{sid}/run", json={"experiment_id": "feature_drift"}).status_code)
    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert codes.count(200) == 1 and all(c in (200, 409) for c in codes)
    runs = client.get(f"{API}/sessions/{sid}").json()["runs"]
    assert sum(r["experiment_id"] == "feature_drift" and r["status"] == "completed" for r in runs) == 1


def test_risky_experiment_needs_explicit_approval(client, demo_project):
    sid = start_session(client, demo_project["id"])["id"]
    r = client.post(f"{API}/sessions/{sid}/run", json={"experiment_id": "retrain_recent"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "approval_required"
    assert client.post(f"{API}/sessions/{sid}/run", json={"experiment_id": "retrain_recent", "approve": True}).status_code == 200


def test_budget_is_enforced_and_session_completes(client, demo_project):
    sid = start_session(client, demo_project["id"], budget=1.2, confidence_threshold=1.0)["id"]
    r = client.post(f"{API}/sessions/{sid}/run", json={"experiment_id": "label_noise"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "experiment_not_feasible"
    last = None
    for _ in range(10):
        r = client.post(f"{API}/sessions/{sid}/run", json={})
        if r.status_code != 200:
            break
        last = r.json()["session"]
        if last["status"] != "active":
            break
    s = client.get(f"{API}/sessions/{sid}").json()
    assert s["spent"] <= 1.2 + 1e-9
    assert s["status"] == "completed" and s["stop_code"] in ("budget_insufficient", "no_feasible_experiments", "low_expected_value")


def test_risk_cap_blocks_experiments(client, demo_project):
    sid = start_session(client, demo_project["id"], risk_cap=0.05)["id"]
    r = client.post(f"{API}/sessions/{sid}/run", json={"experiment_id": "pipeline_replay"})
    assert r.status_code == 409
    nxt = client.get(f"{API}/sessions/{sid}/next").json()
    blocked = next(e for e in nxt["experiments"] if e["id"] == "pipeline_replay")
    assert not blocked["feasible"] and "risk" in blocked["reason_codes"]


def test_external_outcome_and_skip(client, demo_project):
    sid = start_session(client, demo_project["id"])["id"]
    before = client.get(f"{API}/sessions/{sid}").json()
    bad = client.post(f"{API}/sessions/{sid}/external", json={"experiment_id": "feature_drift", "outcome": "banana"})
    assert bad.status_code == 422
    ok = client.post(f"{API}/sessions/{sid}/external", json={"experiment_id": "feature_drift", "outcome": "severe_shift", "note": "ran offline"})
    assert ok.status_code == 200 and ok.json()["run"]["source"] == "external"
    after = ok.json()["session"]
    p0 = next(m for m in before["marginals"] if m["hypothesis"] == "covariate_drift")["probability"]
    p1 = next(m for m in after["marginals"] if m["hypothesis"] == "covariate_drift")["probability"]
    assert p1 > p0
    dup = client.post(f"{API}/sessions/{sid}/external", json={"experiment_id": "feature_drift", "outcome": 0})
    assert dup.status_code == 409
    sk = client.post(f"{API}/sessions/{sid}/skip", json={"experiment_id": "label_noise"})
    assert sk.status_code == 200 and sk.json()["run"]["status"] == "skipped"
    nxt = client.get(f"{API}/sessions/{sid}/next").json()
    assert next(e for e in nxt["experiments"] if e["id"] == "label_noise")["state"] == "skipped"
    assert nxt["selected"] != "label_noise"


def test_missing_labels_makes_experiments_unavailable(client):
    p = new_project(client)
    ref, rec = make_frames(), make_frames(seed=2, shift=1.5)
    upload(client, p["id"], "reference", ref)
    upload(client, p["id"], "recent", rec.drop(columns=["target"]).assign(target=float("nan")))
    assert client.put(f"{API}/projects/{p['id']}/mapping", json={"target": "target", "prediction": "prediction"}).status_code == 200
    s = start_session(client, p["id"])
    assert "labels_recent" in s["symptoms"]["missing_inputs"]
    nxt = s["recommendation"]
    cls = next(e for e in nxt["experiments"] if e["id"] == "class_prior")
    assert not cls["feasible"] and "prereq" in cls["reason_codes"]
    assert next(e for e in nxt["experiments"] if e["id"] == "feature_drift")["feasible"]


def test_session_setup_validation(client, demo_project):
    pid = demo_project["id"]
    r = client.post(f"{API}/sessions", json={"project_id": pid, "hypotheses": [{"id": "covariate_drift"}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "too_few_hypotheses"
    r = client.post(f"{API}/sessions", json={"project_id": pid, "hypotheses": [{"id": "bogus"}, {"id": "label_noise"}]})
    assert r.status_code == 422
    r = client.post(f"{API}/sessions", json={"project_id": pid, "experiments": ["nope"]})
    assert r.status_code == 422
    r = client.post(f"{API}/sessions", json={"project_id": pid, "planner_variant": "magic"})
    assert r.status_code == 422
    r = client.post(f"{API}/sessions", json={"project_id": pid, "risk_cap": 2})
    assert r.status_code == 422


def test_custom_priors_and_planner_variant(client, demo_project):
    s = start_session(client, demo_project["id"], planner_variant="lookahead2", max_active=2,
                      hypotheses=[{"id": "covariate_drift", "prior": 0.5}, {"id": "pipeline_change", "prior": 0.4},
                                  {"id": "label_noise", "prior": 0.1}])
    m = {x["hypothesis"]: x["probability"] for x in s["marginals"]}
    assert set(m) == {"covariate_drift", "pipeline_change", "label_noise"} and s["planner"] == "lookahead2"
    assert m["covariate_drift"] > m["label_noise"]


def test_delete_project_removes_files(client, demo_project, tmp_path):
    pid = demo_project["id"]
    folder = tmp_path / "data" / "projects" / pid
    assert any(folder.iterdir())
    assert client.delete(f"{API}/projects/{pid}").status_code == 204
    assert not folder.exists() or not any(folder.iterdir())


def test_dataset_cannot_be_replaced_while_sessions_exist(client, demo_project):
    start_session(client, demo_project["id"])
    r = upload(client, demo_project["id"], "reference", make_frames())
    assert r.status_code == 409


# ---------------------------------------------------------------- benchmark lab
def test_benchmark_lab_job_runs_and_exports(client):
    import time
    body = {"name": "tiny", "datasets": ["synthetic_credit"], "faults": ["covariate_drift", "metric_bug"], "severities": ["high"],
            "planners": ["random", "adaptive"], "seeds": [900], "budgets": [2, 6]}
    r = client.post(f"{API}/benchmarks", json=body)
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    for _ in range(240):
        st = client.get(f"{API}/benchmarks/{jid}").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert st["status"] == "completed", st
    res = client.get(f"{API}/benchmarks/{jid}/results").json()
    assert res["simulated"] is True and {s["planner"] for s in res["summary"]} == {"random", "adaptive"}
    assert client.get(f"{API}/benchmarks/{jid}/export.csv").text.startswith("scenario")
    assert client.get(f"{API}/benchmarks/{jid}/traces").json()["total"] > 0
    assert client.get(f"{API}/benchmarks/{jid}/export.json").status_code == 200


def test_benchmark_rejects_seeds_used_for_estimation(client):
    r = client.post(f"{API}/benchmarks", json={"datasets": ["synthetic_credit"], "seeds": [0, 1]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "seed_overlap"
    assert client.post(f"{API}/benchmarks", json={"datasets": ["nope"]}).status_code == 422
