"""Background runs for pack agents on AgentCore Runtime.

Pattern (AWS, "Handle asynchronous and long running agents with Amazon Bedrock
AgentCore Runtime", read 2026-09-23): register a task with app.add_async_task before
the work starts, do the work off the entrypoint thread, call app.complete_async_task
when it ends. While a task is registered the SDK answers the runtime's /ping with
HealthyBusy and the session lives past the 15-minute idle limit, up to 8 hours.
The entrypoint itself returns in seconds, which keeps it under the 15-minute
synchronous request cap.

Two things the AWS example does not cover that matter here:

1. The Graph identity token (tools/graph_identity.py) is read from a context variable
   that AgentCore sets per request. A new thread does not inherit context variables,
   so the worker runs under a copy of the entrypoint's context (contextvars.copy_context).
2. The caller has hung up by the time the work finishes, so results go to S3
   (tools/s3_store.py), and a second invoke with mode "status" reads them back.

An agent takes part by providing two functions:
    prepare(inp) -> dict     foreground, seconds: validate, list inputs, return facts
                             for the status document (e.g. files_total). Raises on a
                             bad input so the caller gets the error, not a running job.
    work(inp, progress) -> {"output": ..., "usage": ...}
                             background, as long as it takes. progress(dict) merges
                             fields into status.json as the run moves.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from tools import s3_store

log = logging.getLogger("jonbatiste.jobs")

STATES = ("running", "done", "done_with_errors", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _keys(agent: str, inp: dict) -> dict:
    prefix = s3_store.run_prefix(agent, inp["client_id"], inp["run_id"])
    return {"prefix": prefix, "status": f"{prefix}/status.json",
            "response": f"{prefix}/response.json", "markdown": f"{prefix}/response.md"}


def start(app, agent: str, inp: dict, prepare: Callable[[dict], dict],
          work: Callable[[dict, Callable[[dict], None]], dict]) -> dict:
    """Foreground: prepare, write the first status, register the task, spawn the worker, return."""
    if not s3_store.enabled():
        raise RuntimeError(f"mode 'start' needs {s3_store.BUCKET_ENV} on the runtime; "
                           "use mode 'sync' for laptop runs")
    keys = _keys(agent, inp)
    facts = prepare(inp)  # raises to the caller on a bad folder, missing grant, etc.

    status: dict[str, Any] = {
        "agent": agent, "client_id": inp["client_id"], "run_id": inp["run_id"],
        "state": "running", "started_at": _now(), "updated_at": _now(),
        "bucket": s3_store.bucket(), "keys": keys, **facts,
    }
    lock = threading.Lock()

    def write_status() -> None:
        status["updated_at"] = _now()
        s3_store.put_json(keys["status"], status)

    def progress(fields: dict) -> None:
        with lock:
            status.update(fields)
            write_status()

    write_status()
    task_id = app.add_async_task(f"{agent}:{inp['run_id']}", {"client_id": inp["client_id"]})

    def worker() -> None:
        try:
            result = work(inp, progress)
            out = result["output"]
            s3_store.put_json(keys["response"], {"agent": agent, "output": out, "usage": result["usage"]})
            if out.get("markdown"):
                s3_store.put_text(keys["markdown"], out["markdown"])
            with lock:
                status.pop("partial", None)   # the full result is in response.json now
                status["state"] = "done_with_errors" if out.get("files_failed") else "done"
                status["usage"] = result["usage"]
                status["summary"] = _summary(out)
                write_status()
        except Exception as e:  # noqa: BLE001  the status document is the error channel
            log.exception("%s run %s failed", agent, inp["run_id"])
            with lock:
                partial = status.pop("partial", None)
                if partial:
                    # Extraction finished before the failure: keep the rows, do not lose the spend.
                    s3_store.put_json(f"{keys['prefix']}/partial.json", partial)
                    status["partial_key"] = f"{keys['prefix']}/partial.json"
                status["state"] = "failed"
                status["error"] = f"{type(e).__name__}: {e}"
                status["traceback"] = traceback.format_exc()[-4000:]
                write_status()
        finally:
            app.complete_async_task(task_id)

    ctx = contextvars.copy_context()
    threading.Thread(target=ctx.run, args=(worker,), daemon=True,
                     name=f"{agent}-{inp['run_id']}").start()
    return {"agent": agent, "run_id": inp["run_id"], "state": "running",
            "bucket": s3_store.bucket(), "keys": keys, **facts}


def status(agent: str, inp: dict) -> dict:
    """Read status.json for a run. Unknown run returns state 'unknown', not an error."""
    keys = _keys(agent, inp)
    doc = s3_store.get_json(keys["status"])
    if doc is None:
        return {"agent": agent, "run_id": inp["run_id"], "state": "unknown", "keys": keys}
    return doc


def _summary(out: dict) -> dict:
    """Counts only. The full result is in response.json."""
    s: dict[str, Any] = {}
    for k in ("ledger", "sources_read", "next_best_questions", "contradictions", "files_failed"):
        if isinstance(out.get(k), list):
            s[k] = len(out[k])
    if isinstance(out.get("gate"), dict):
        s["gate_passed"] = out["gate"].get("passed")
        s["gate_failing"] = len(out["gate"].get("failing", []))
    return s
