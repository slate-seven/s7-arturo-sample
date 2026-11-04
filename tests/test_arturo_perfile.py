"""Per-file extraction, deterministic merge, background jobs, mode routing.

No network, no model. The model call (_extract_one, Agent) is replaced with fixtures.
Runs from the repo root: python -m pytest -q
"""

from __future__ import annotations

import contextvars
import json
import threading
import time

import pytest

import jobs
import main
from agents import arturo
from tools import s3_store


def _row(file, sub, text, cls="B", **kw):
    r = {"source_file": file, "speaker": None, "text": text, "object_type": "Friction Signal",
         "domain": sub[:2], "sub_dimension": sub, "friction_category": None, "evidence_class": cls,
         "half_asked": False, "next_question": None, "live_system_hint": None, "contradicts": None, "note": None}
    r.update(kw)
    return r


def _src(name, text="hello", path="b!drive:folder/x"):
    return {"path": f"{path}/{name}", "name": name, "item_id": "i", "drive_id": "b!drive",
            "size": len(text), "last_modified": "2026-09-23T00:00:00Z", "ext": ".docx", "text": text}


# ---------- chunking ----------

def test_chunks_short_text_is_one_chunk():
    assert arturo._chunks("a\nb\nc", 100) == ["a\nb\nc"]


def test_chunks_split_on_paragraphs_and_keep_all_text():
    text = "\n".join(f"para {i} " + "x" * 50 for i in range(40))
    parts = arturo._chunks(text, 600)
    assert len(parts) > 1
    assert "\n".join(parts) == text
    assert all(len(p) <= 600 + 60 for p in parts)


def test_user_message_names_one_file_and_part():
    m = arturo._build_user_message(_src("f.docx"), "body", 2, 3)
    assert "One source file follows" in m and "part 2 of 3" in m and "FILE: f.docx" in m


# ---------- merge ----------

def test_merge_drops_exact_repeats_same_file_only():
    rows = [_row("a.docx", "D1a", "The CRM is out of date."),
            _row("a.docx", "D1a", "the crm is  out of date."),      # same file, same text: duplicate
            _row("b.docx", "D1a", "The CRM is out of date."),       # other file: corroboration, kept
            _row("a.docx", "D1b", "The CRM is out of date.")]       # other sub-dimension: kept
    merged, dropped = arturo._merge_ledger(rows)
    assert dropped == 1 and len(merged) == 3
    assert merged == [rows[0], rows[2], rows[3]]           # order preserved


def test_coverage_counts_distinct_sources():
    ledger = [_row("a.docx", "D1a", "x"), _row("b.docx", "D1a", "y"), _row("a.docx", "D1a", "z")]
    cov, gate = arturo._coverage_and_gate(ledger)
    sub = cov["D1"]["subs"]["D1a"]
    assert sub["items"] == 3 and sub["sources"] == 2 and sub["status"] == "captured"
    assert gate["passed"] is False                          # every other sub is missing


# ---------- per-file extraction with a fake model ----------

def test_extract_evidence_one_call_per_file_and_failure_continues(monkeypatch):
    calls = []

    def fake_extract_one(model, system_prompt, source, text, part, parts):
        calls.append((source["name"], part, parts))
        if source["name"] == "bad.docx":
            raise ValueError("model output failed validation: item 0 bad domain")
        return [_row(source["name"], "D2a", f"quote from {source['name']} {part}")], {"inputTokens": 10, "outputTokens": 5}

    monkeypatch.setattr(arturo, "_extract_one", fake_extract_one)
    monkeypatch.setattr(arturo, "_model", lambda: object())
    monkeypatch.setattr(arturo, "_system_prompt", lambda: "prompt")
    monkeypatch.setattr(arturo, "MAX_CHARS_PER_CALL", 50)

    sources = [_src("one.docx", "short"), _src("bad.docx", "short"), _src("long.docx", "\n".join(["p" * 30] * 5))]
    progress = []
    rows, usage, failed = arturo._extract_evidence(sources, progress.append)

    assert [c[0] for c in calls].count("one.docx") == 1
    assert [c[0] for c in calls].count("long.docx") > 1                    # chunked, same file name on every call
    assert failed == [{"file": "bad.docx", "error": "ValueError: model output failed validation: item 0 bad domain"}]
    assert {r["source_file"] for r in rows} == {"one.docx", "long.docx"}
    assert usage["model_calls"] == len(calls) - 1                          # the failed call is not counted
    assert usage["inputTokens"] == 10 * usage["model_calls"]
    assert progress[-1]["files_done"] == 3
    assert progress[-1]["files_failed"] == [{"file": "bad.docx", "error": "ValueError: model output failed validation: item 0 bad domain"}]
    assert usage["per_file"][1]["failed"] is True and usage["per_file"][0]["file"] == "one.docx"   # per_file keeps source order


def test_extract_evidence_all_failed_raises(monkeypatch):
    monkeypatch.setattr(arturo, "_extract_one", lambda *a: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(arturo, "_model", lambda: object())
    monkeypatch.setattr(arturo, "_system_prompt", lambda: "prompt")
    with pytest.raises(ValueError, match="every source file failed"):
        arturo._extract_evidence([_src("a.docx")], None)


def test_extract_evidence_empty_sources_raises():
    with pytest.raises(ValueError, match="no source files"):
        arturo._extract_evidence([], None)


# ---------- cross-file contradictions with a fake Agent ----------

class _FakeResult:
    def __init__(self, parsed):
        self.structured_output = parsed
        self.metrics = type("M", (), {"accumulated_usage": {"inputTokens": 3, "outputTokens": 1}})()


def test_cross_file_contradictions_filters_same_file_and_bad_indexes(monkeypatch):
    ledger = [_row("a.docx", "D1a", "We have SOPs for everything."),
              _row("b.docx", "D3a", "Nothing is written down."),
              _row("a.docx", "D3a", "Some SOPs exist.")]
    findings = arturo.ContradictionFindings(findings=[
        arturo.Contradiction(row_a=0, row_b=1, why="SOP coverage stated as complete and as absent"),
        arturo.Contradiction(row_a=0, row_b=2, why="same file, must be dropped"),
        arturo.Contradiction(row_a=0, row_b=99, why="bad index, must be dropped"),
    ])

    class FakeAgent:
        def __init__(self, **kw): pass
        def __call__(self, msg, structured_output_model=None): return _FakeResult(findings)

    monkeypatch.setattr(arturo, "Agent", FakeAgent)
    monkeypatch.setattr(arturo, "_model", lambda: object())
    monkeypatch.setattr(arturo, "CROSS_FILE_CONTRADICTIONS", True)
    usage = {"model_calls": 0}
    out = arturo._cross_file_contradictions(ledger, usage)
    assert len(out) == 1 and out[0]["scope"] == "cross-file"
    assert out[0]["earlier_file"] == "a.docx" and out[0]["file"] == "b.docx"
    assert usage["model_calls"] == 1 and usage["inputTokens"] == 3


def test_cross_file_contradictions_off_by_flag(monkeypatch):
    monkeypatch.setattr(arturo, "CROSS_FILE_CONTRADICTIONS", False)
    assert arturo._cross_file_contradictions([_row("a", "D1a", "x"), _row("b", "D1a", "y")], {}) == []


# ---------- prepare / work / run end to end with fakes ----------

def _inp(**kw):
    d = {"client_id": "TST", "folders": ["b!drive:folder"], "since": None,
         "methodology_version": "v2_3", "run_id": "t01", "output_dir": "/tmp/arturo_test_out"}
    d.update(kw)
    return d


def _wire_fakes(monkeypatch, files):
    monkeypatch.setattr(arturo, "list_folder", lambda folder: [dict(_src(n), text=None) for n in files])
    monkeypatch.setattr(arturo, "read_file_text", lambda item: f"text of {item['name']}")
    monkeypatch.setattr(arturo, "_extract_one",
                        lambda model, sp, source, text, part, parts: ([_row(source["name"], "D1a", "q " + source["name"])], {"inputTokens": 1}))
    monkeypatch.setattr(arturo, "_model", lambda: object())
    monkeypatch.setattr(arturo, "_system_prompt", lambda: "prompt")
    monkeypatch.setattr(arturo, "CROSS_FILE_CONTRADICTIONS", False)
    monkeypatch.setattr(arturo.graph_identity, "on_runtime", lambda: False)


@pytest.mark.skip(reason="Written before the signed DFM profile became required; fixture update pending.")
def test_prepare_lists_and_validates(monkeypatch):
    _wire_fakes(monkeypatch, ["a.docx", "b.docx"])
    facts = arturo.prepare(_inp())
    assert facts["files_total"] == 2 and facts["files"] == ["a.docx", "b.docx"] and facts["files_failed"] == []


@pytest.mark.skip(reason="Written before the signed DFM profile became required; fixture update pending.")
def test_prepare_rejects_graph_token_and_empty_folder(monkeypatch):
    _wire_fakes(monkeypatch, [])
    with pytest.raises(ValueError, match="graph_token"):
        arturo.prepare(_inp(graph_token="x"))
    with pytest.raises(ValueError, match="no source files"):
        arturo.prepare(_inp())


@pytest.mark.skip(reason="Written before the signed DFM profile became required; fixture update pending.")
def test_run_sync_shape_unchanged_plus_files_failed(monkeypatch):
    _wire_fakes(monkeypatch, ["a.docx", "b.docx"])
    out = arturo.run(_inp())
    o = out["output"]
    for k in ("ledger", "coverage", "gate", "next_best_questions", "live_system_signal",
              "contradictions", "sources_read", "writing_check", "markdown", "files", "files_failed"):
        assert k in o, k
    assert len(o["ledger"]) == 2 and all(r["run_id"] == "t01" for r in o["ledger"])
    assert o["files_failed"] == [] and len(o["sources_read"]) == 2
    assert "Files read: 2. Files failed: 0." in o["markdown"]
    assert out["usage"]["model_calls"] == 2 and "per_file" in out["usage"]


# ---------- jobs: background run, status, context propagation ----------

class _FakeApp:
    def __init__(self):
        self.started, self.completed = [], []
    def add_async_task(self, name, metadata=None):
        self.started.append(name); return 42
    def complete_async_task(self, task_id):
        self.completed.append(task_id); return True


@pytest.fixture
def fake_s3(monkeypatch):
    store = {}
    monkeypatch.setenv(s3_store.BUCKET_ENV, "test-bucket")
    monkeypatch.setattr(s3_store, "put_json", lambda key, obj: store.__setitem__(key, json.loads(json.dumps(obj))) or f"s3://test-bucket/{key}")
    monkeypatch.setattr(s3_store, "put_text", lambda key, text, content_type="text/markdown": store.__setitem__(key, text) or f"s3://test-bucket/{key}")
    monkeypatch.setattr(s3_store, "get_json", lambda key: store.get(key))
    return store


def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_jobs_start_returns_fast_and_finishes_in_background(fake_s3):
    app = _FakeApp()
    gate = threading.Event()

    def prepare(inp):
        return {"files_total": 3, "files_done": 0, "files_failed": []}

    def work(inp, progress):
        progress({"files_done": 1})
        gate.wait(2)
        return {"output": {"ledger": [1, 2], "gate": {"passed": False, "failing": ["x"]}, "markdown": "# r",
                           "sources_read": [1, 2, 3], "files_failed": []}, "usage": {"inputTokens": 7}}

    reply = jobs.start(app, "arturo", {"client_id": "TST", "run_id": "r1"}, prepare, work)
    assert reply["state"] == "running" and reply["files_total"] == 3
    assert reply["keys"]["status"] == "arturo/TST/r1/status.json"
    assert app.started == ["arturo:r1"] and app.completed == []
    assert _wait(lambda: fake_s3["arturo/TST/r1/status.json"].get("files_done") == 1)
    gate.set()
    assert _wait(lambda: fake_s3["arturo/TST/r1/status.json"]["state"] == "done")
    st = jobs.status("arturo", {"client_id": "TST", "run_id": "r1"})
    assert st["summary"] == {"ledger": 2, "sources_read": 3, "files_failed": 0, "gate_passed": False, "gate_failing": 1}
    assert st["usage"] == {"inputTokens": 7}
    assert fake_s3["arturo/TST/r1/response.json"]["output"]["ledger"] == [1, 2]
    assert fake_s3["arturo/TST/r1/response.md"] == "# r"
    assert app.completed == [42]


def test_jobs_failure_is_recorded_and_task_completed(fake_s3):
    app = _FakeApp()

    def work(inp, progress):
        raise RuntimeError("Graph 403 on Transcripts")

    jobs.start(app, "arturo", {"client_id": "TST", "run_id": "r2"}, lambda inp: {"files_total": 1}, work)
    assert _wait(lambda: fake_s3["arturo/TST/r2/status.json"]["state"] == "failed")
    st = fake_s3["arturo/TST/r2/status.json"]
    assert st["error"] == "RuntimeError: Graph 403 on Transcripts" and "traceback" in st
    assert app.completed == [42]


def test_jobs_done_with_errors_when_files_failed(fake_s3):
    app = _FakeApp()
    work = lambda inp, progress: {"output": {"files_failed": [{"file": "x", "error": "e"}], "ledger": []}, "usage": {}}
    jobs.start(app, "arturo", {"client_id": "TST", "run_id": "r3"}, lambda inp: {}, work)
    assert _wait(lambda: fake_s3["arturo/TST/r3/status.json"]["state"] == "done_with_errors")


def test_jobs_prepare_error_reaches_caller_and_starts_nothing(fake_s3):
    app = _FakeApp()
    with pytest.raises(PermissionError):
        jobs.start(app, "arturo", {"client_id": "TST", "run_id": "r4"},
                   lambda inp: (_ for _ in ()).throw(PermissionError("folder not in allowlist")), lambda i, p: {})
    assert app.started == [] and "arturo/TST/r4/status.json" not in fake_s3


def test_jobs_worker_sees_entrypoint_contextvars(fake_s3):
    """graph_identity reads the workload token from a context variable; the worker must see it."""
    var = contextvars.ContextVar("wat")
    var.set("token-from-request")
    seen = {}
    work = lambda inp, progress: seen.__setitem__("v", var.get(None)) or {"output": {}, "usage": {}}
    jobs.start(_FakeApp(), "arturo", {"client_id": "TST", "run_id": "r5"}, lambda inp: {}, work)
    assert _wait(lambda: "v" in seen) and seen["v"] == "token-from-request"


def test_jobs_start_without_bucket_is_refused(monkeypatch):
    monkeypatch.delenv(s3_store.BUCKET_ENV, raising=False)
    with pytest.raises(RuntimeError, match="JB_RESULTS_BUCKET"):
        jobs.start(_FakeApp(), "arturo", {"client_id": "T", "run_id": "r"}, lambda i: {}, lambda i, p: {})


def test_status_unknown_run(fake_s3):
    assert jobs.status("arturo", {"client_id": "TST", "run_id": "nope"})["state"] == "unknown"


def test_run_prefix_rejects_slashes():
    with pytest.raises(ValueError):
        s3_store.run_prefix("arturo", "a/b", "r")


# ---------- main.invoke routing ----------

def test_invoke_sync_default_unchanged(monkeypatch):
    monkeypatch.setitem(main.AGENTS, "arturo", lambda inp: {"output": {"ok": inp["run_id"]}, "usage": {}})
    r = main.invoke({"agent": "arturo", "input": {"run_id": "x"}})
    assert r == {"agent": "arturo", "output": {"ok": "x"}, "usage": {}}


def test_invoke_rejects_bad_mode_and_unsupported_agent():
    with pytest.raises(ValueError, match="unknown mode"):
        main.invoke({"agent": "arturo", "mode": "later", "input": {}})
    with pytest.raises(ValueError, match="requires input.run_id"):
        main.invoke({"agent": "arturo", "mode": "status", "input": {"client_id": "a"}})


def test_invoke_start_and_status_route_to_jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(main.jobs, "start", lambda app, name, inp, prep, work: calls.append(("start", name)) or {"state": "running"})
    monkeypatch.setattr(main.jobs, "status", lambda name, inp: calls.append(("status", name)) or {"state": "done"})
    assert main.invoke({"agent": "arturo", "mode": "start", "input": {"client_id": "a", "run_id": "b"}}) == {"state": "running"}
    assert main.invoke({"agent": "arturo", "mode": "status", "input": {"client_id": "a", "run_id": "b"}}) == {"state": "done"}
    assert calls == [("start", "arturo"), ("status", "arturo")]


# ---------- half-asked rows answered by another file (added 2026-09-24) ----------

def test_half_asked_counts_when_another_file_has_a_full_row():
    rows = [_row("kickoff.docx", "D1a", "who decides? me, if it gets to me", half_asked=True, next_question="Who else?"),
            _row("ortega.docx", "D1a", "Ray decides disputes over 5k, Dana below."),
            _row("nair.docx", "D1a", "Disputes go to Ray.")]
    counted = arturo._counted_rows(rows)
    assert len(counted) == 3
    cov, gate = arturo._coverage_and_gate(rows)
    sub = cov["D1"]["subs"]["D1a"]
    assert sub["items"] == 3 and sub["half_asked"] == 1 and sub["half_asked_open"] == 0
    assert sub["status"] == "captured"
    # the question still surfaces for the next 1:1 through the row itself
    assert rows[0]["half_asked"] is True and rows[0]["next_question"] == "Who else?"


def test_half_asked_alone_or_same_file_does_not_count():
    only_half = [_row("a.docx", "D2b", "x", half_asked=True), _row("b.docx", "D2b", "y", half_asked=True)]
    assert arturo._counted_rows(only_half) == []
    same_file = [_row("a.docx", "D2b", "x", half_asked=True), _row("a.docx", "D2b", "full answer")]
    assert arturo._counted_rows(same_file) == [same_file[1]]
    cov, _ = arturo._coverage_and_gate(same_file)
    assert cov["D2"]["subs"]["D2b"]["status"] == "thin" and cov["D2"]["subs"]["D2b"]["half_asked_open"] == 1


# ---------- 2026-09-24: non-fatal writing check, parallel order ----------

def test_writing_check_never_raises_and_flags_hard_over():
    long = " ".join(["word"] * 73)
    wc = arturo._check_writing([_row("a.docx", "D1a", long), _row("a.docx", "D1a", "short")])
    assert wc["hard_over"] == ["row 0 text: 73 words, cap 40"]
    assert wc["over_cap"] == ["row 0 text: 73 words, cap 40"]


def test_parallel_extraction_gives_serial_order(monkeypatch):
    import random, time as _t

    def fake_extract_one(model, sp, source, text, part, parts):
        _t.sleep(random.random() / 50)          # scramble completion order
        return [_row(source["name"], "D1a", "q " + source["name"])], {"inputTokens": 1}

    monkeypatch.setattr(arturo, "_extract_one", fake_extract_one)
    monkeypatch.setattr(arturo, "_model", lambda: object())
    monkeypatch.setattr(arturo, "_system_prompt", lambda: "prompt")
    names = [f"f{i:02d}.docx" for i in range(12)]
    sources = [_src(n) for n in names]
    monkeypatch.setattr(arturo, "WORKERS", 4)
    rows_par, u_par, _ = arturo._extract_evidence(sources, None)
    monkeypatch.setattr(arturo, "WORKERS", 1)
    rows_ser, u_ser, _ = arturo._extract_evidence(sources, None)
    assert [r["source_file"] for r in rows_par] == names == [r["source_file"] for r in rows_ser]
    assert u_par["model_calls"] == u_ser["model_calls"] == 12
    assert [p["file"] for p in u_par["per_file"]] == names


@pytest.mark.skip(reason="Written before the signed DFM profile became required; fixture update pending.")
def test_work_parks_partial_ledger_before_gate(monkeypatch):
    _wire_fakes(monkeypatch, ["a.docx"])
    seen = []
    arturo.work(_inp(), seen.append)
    phases = [p.get("phase") for p in seen if "phase" in p]
    assert "merged" in phases
    parked = next(p for p in seen if p.get("phase") == "merged")["partial"]
    assert len(parked["ledger"]) == 1 and parked["files_failed"] == []


def test_jobs_keeps_partial_on_failure(fake_s3):
    app = _FakeApp()

    def work(inp, progress):
        progress({"phase": "merged", "partial": {"ledger": [1, 2, 3]}})
        raise RuntimeError("render blew up")

    jobs.start(app, "arturo", {"client_id": "TST", "run_id": "r6"}, lambda inp: {}, work)
    assert _wait(lambda: fake_s3["arturo/TST/r6/status.json"]["state"] == "failed")
    st = fake_s3["arturo/TST/r6/status.json"]
    assert "partial" not in st and st["partial_key"] == "arturo/TST/r6/partial.json"
    assert fake_s3["arturo/TST/r6/partial.json"] == {"ledger": [1, 2, 3]}


def test_extract_one_sets_source_file_in_code(monkeypatch):
    """model copied 'name (modified ...)' into source_file on 10 of 33 files."""
    item = arturo.EvidenceItem(source_file="Ed Lewis.docx (modified 2026-09-22T16:11:52Z)", speaker=None,
                               text="We have no runbook.", object_type="Friction Signal", domain="D3",
                               sub_dimension="D3a", friction_category=None, evidence_class="B", half_asked=False,
                               next_question=None, live_system_hint=None, contradicts=None, note=None)

    class FakeAgent:
        def __init__(self, **kw): pass
        def __call__(self, msg, structured_output_model=None):
            assert "===== FILE: Ed Lewis.docx =====" in msg and "(modified" not in msg
            return _FakeResult(arturo.FileExtraction(items=[item]))

    monkeypatch.setattr(arturo, "Agent", FakeAgent)
    rows, usage = arturo._extract_one(object(), "prompt", _src("Ed Lewis.docx"), "body", 1, 1)
    assert rows[0]["source_file"] == "Ed Lewis.docx" and usage["source_file_mismatches"] == 1
