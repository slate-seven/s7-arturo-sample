"""Result store for pack runs: one S3 prefix per run.

Why this exists (2026-09-23): a synchronous AgentCore invoke is capped at 15 minutes
(AgentCore Runtime limits table). A background run can go to 8 hours, but by then the
caller has hung up, so the result has to land somewhere durable that the caller and,
later, Katie's trigger can read. The microVM disk does not qualify: a stopped session
gets a fresh machine on the next invoke.

Layout:
    s3://<bucket>/<agent>/<client_id>/<run_id>/status.json     small, rewritten as the run moves
    s3://<bucket>/<agent>/<client_id>/<run_id>/response.json   full result, same shape as the sync reply
    s3://<bucket>/<agent>/<client_id>/<run_id>/response.md     the report
    s3://<bucket>/<agent>/<client_id>/<run_id>/raw/...         model replies that failed validation

Bucket comes from JB_RESULTS_BUCKET, set on the runtime by the CDK stack. With no bucket
set, writes are skipped and the caller is told (sync laptop runs do not need S3).
Only PutObject and GetObject are used; the runtime role gets nothing else.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

BUCKET_ENV = "JB_RESULTS_BUCKET"
REGION = os.environ.get("AWS_REGION", "us-east-1")

_client = None


def bucket() -> str | None:
    return os.environ.get(BUCKET_ENV) or None


def enabled() -> bool:
    return bucket() is not None


def _s3():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=REGION)
    return _client


def run_prefix(agent: str, client_id: str, run_id: str) -> str:
    for part in (agent, client_id, run_id):
        if not part or "/" in part or ".." in part:
            raise ValueError(f"bad key part {part!r}")
    return f"{agent}/{client_id}/{run_id}"


def put_json(key: str, obj: Any) -> str | None:
    b = bucket()
    if b is None:
        return None
    _s3().put_object(Bucket=b, Key=key, Body=json.dumps(obj, indent=2).encode("utf-8"),
                     ContentType="application/json")
    return f"s3://{b}/{key}"


def put_text(key: str, text: str, content_type: str = "text/markdown") -> str | None:
    b = bucket()
    if b is None:
        return None
    _s3().put_object(Bucket=b, Key=key, Body=text.encode("utf-8"), ContentType=content_type)
    return f"s3://{b}/{key}"


def get_json(key: str) -> Any | None:
    """Return the parsed object, or None when the key does not exist."""
    b = bucket()
    if b is None:
        raise RuntimeError(f"{BUCKET_ENV} is not set; nothing to read from")
    try:
        r = _s3().get_object(Bucket=b, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(r["Body"].read().decode("utf-8"))


def get_text(key: str) -> str | None:
    """Return the object as UTF-8 text, or None when the key does not exist. Added 2026-10-11 for
    first-system documents in the client folder."""
    b = bucket()
    if b is None:
        raise RuntimeError(f"{BUCKET_ENV} is not set; nothing to read from")
    try:
        r = _s3().get_object(Bucket=b, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return r["Body"].read().decode("utf-8")
