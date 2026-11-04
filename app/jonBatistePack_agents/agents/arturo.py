"""Arturo S.: Methodology and Evidence.

Reads transcripts, field notes and lists from the SharePoint roots in the client's source map,
classifies each evidence item against Clarity Sprint Methodology v2_3
(evidence classes A to F, domains D1 to D5, five sub-dimensions each),
and reports what is captured, missing, or half asked.

Facts are decided in code before the model runs. The model extracts and
classifies evidence items; Python counts them against the minimums and
sets the gate. The model never decides whether there is "enough evidence".

Extraction is one model call per source file (2026-09-23). A 9/23 client run put
33 files in one request and the Bedrock stream sat silent for 900 seconds. Per-file
calls keep each request small, give every row a single provenance, and make the
stability run meaningful. Rows are merged in code: exact duplicates dropped, source
count per sub-dimension kept as corroboration. One small second call reads the merged
rows for contradictions across files; everything else downstream is arithmetic.

On AgentCore the run goes through jobs.py (mode "start" / "status") so it can exceed
the 15-minute synchronous cap; run() stays synchronous for laptop runs and tests.

Input contract (all keys required unless marked optional):
    client_id            str   e.g. "TST"
    since                str|None  ISO date; skip files modified before it. Optional.
    methodology_version  str   must be "v2_3" for this build
    run_id               str   caller-supplied; goes into every output row
    output_dir           str   local path for the two output files
    profile_path         str   optional, laptop runs only: local path to the signed DFM profile JSON
    source_map_path      str   optional, laptop runs only: local path to the source map JSON
A "folders" key is rejected (2026-10-11): the roots come from the source map, not the payload.

Signed DFM (2026-10-11, Chris, portability review R08): Arturo reads nothing until
profile/<client_id>/dfm_profile.json is in the run bucket. The file's presence is proof the
Data Flow Map is signed; without it the run stops before any SharePoint read, same as Fe.

SharePoint credential: on AgentCore Runtime, Arturo's own identity via AgentCore Identity
(tools/graph_identity.py). On a laptop, GRAPH_TOKEN in the environment. A graph_token key in
the payload is rejected (Chris ruling 9/20: payload tokens were test only).

Source map (2026-10-09; complete root list 2026-10-11, portability review R01 to R03, R06, R09):
profile/<client_id>/source_map.json in the run bucket, next to the DFM profile. It is the
complete list of SharePoint roots Arturo reads: every root in it, nothing else. Keyed by the
root string. Each entry carries:
    provider       str        required: "s7" (Slate Seven app registration) or "client" (the
                              client-tenant app registration); picks the Graph token
    instructions   str        optional; appended to the extraction message for files from that root only
    drop_columns   list[str]  optional; list roots: columns left out of the rendered table, in code
    source_kind    str        optional; stamped on every ledger row from that root (e.g. "ai_roster")
    skip_folders   list[str]  optional; folder names under that root not walked, on top of the
                              standing skips in sharepoint_read (Archive, Agent Pack Outputs)
A root may be a folder, a single file, or a list (2026-10-12).

Files (2026-10-12, client source list): every file under the roots is listed with a verdict
before anything is read. prepare() returns that list as "plan", so the start response shows what
the run will read and what it will skip, and why. A file that will not open (password protected,
scanned PDF, no reader for its type) is recorded and the run continues; it never stops the run.
Images are transcribed to text by one model call each and read like any other file.
The map sets sharepoint_read's allowlist for the run; Graph Sites.Selected stays the second
gate. A missing, empty or broken map stops the run before any read (Chris 10/11): without it
Arturo has nothing to read. No client value lives in code.
First use: a client's AI roster list, rows typed as existing AI systems. Arturo still extracts and
classifies only; the map tells him what a file is, not what to conclude from it.

Output:
    {
      "output": {
        "ledger":   [...canonical object rows...],
        "coverage": {domain: {sub_dimension: {"status", "items", "weight", ...}}},
        "gate":     {"passed": bool, "failing": [...]},
        "next_best_questions": [...],
        "live_system_signal": [...],
        "files": {"json": path, "md": path},
      },
      "usage": {inputTokens, outputTokens, totalTokens,
                cacheReadInputTokens, cacheWriteInputTokens}
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from typing import Literal, Optional

from botocore.config import Config as BotocoreConfig
from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel, CacheConfig

from tools import graph_identity, s3_store, sharepoint_read
from tools.sharepoint_read import list_folder, read_file_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "arturo_system.md"
WRITING_STANDARD_PATH = REPO_ROOT / "prompts" / "s7_writing_standard.md"
RULES_PATH = REPO_ROOT / "reference" / "methodology_v2_3_evidence_rules.md"

# Word caps on fields the model authors. Enforced in code after generation.
CAP_TEXT = 40           # quoted evidence; the prompt already says 40
CAP_QUESTION = 30       # next-best question for a 1:1
CAP_NOTE = 40
CAP_LABEL = 4           # live_system_hint canonical label
HARD_OVER = 1.5         # a field more than 50% over its cap fails the run

# Words the agent may not author (Slate Seven standard). Quoted source text is exempt.
BANNED_WORDS = {"need", "actually", "real", "really", "just", "delve", "leverage", "robust",
                "seamless", "pivotal", "crucial", "landscape", "journey", "holistic", "synergy",
                "empower", "unlock", "streamline", "utilize"}
AUTHORED_FIELDS = ("next_question", "note", "live_system_hint")

# One inference profile for the pack; override per environment, never per agent file.
MODEL_ID = os.environ.get("JB_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
REGION = os.environ.get("AWS_REGION", "us-east-1")

REQUIRED_INPUT = ("client_id", "methodology_version", "run_id", "output_dir")
METHODOLOGY_VERSION = "v2_3"

SOURCE_MAP_KEYS = {"provider", "instructions", "drop_columns", "source_kind", "skip_folders"}
PROVIDERS = {"s7", "client"}
PROFILE_DIR = "profile"                        # client folder in the run bucket: profile/<client_id>/
DFM_PROFILE_FILE = "dfm_profile.json"          # signed Data Flow Map summary; required


def source_map_key(client_id: str) -> str:
    """S3 key of the client's source map in the run bucket. Same folder as dfm_profile.json."""
    return f"profile/{client_id}/source_map.json"


def _require_dfm(inp: dict) -> str:
    """Stop unless the signed DFM profile is on file. Returns where it was found. Arturo does not
    read the profile's contents; its presence is the gate."""
    path = inp.get("profile_path")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"arturo: profile_path not found: {path}")
        return path
    key = f"{PROFILE_DIR}/{inp['client_id']}/{DFM_PROFILE_FILE}"
    if not s3_store.enabled():
        raise RuntimeError("arturo: no run bucket and no profile_path; the signed DFM profile is required")
    if s3_store.get_json(key) is None:
        raise FileNotFoundError(f"arturo: the signed Data Flow Map is not on file for {inp['client_id']} "
                                f"(s3://{s3_store.bucket()}/{key}); Arturo does not run without it")
    return key


def _load_source_map(inp: dict) -> tuple[dict, dict]:
    """Read and check the source map, then set it as sharepoint_read's allowlist. Returns
    (map, info); info goes into the result. Missing, empty or broken stops the run."""
    client_id = inp["client_id"]
    path = inp.get("source_map_path")
    if path:
        key = path
        try:
            with open(path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"arturo: source_map_path not found: {path}")
        except json.JSONDecodeError as ex:
            raise ValueError(f"arturo: source map {path} is not valid JSON: {ex}")
    else:
        key = source_map_key(client_id)
        if not s3_store.enabled():
            raise RuntimeError("arturo: no run bucket and no source_map_path; the source map lists what to read")
        try:
            raw = s3_store.get_json(key)
        except ValueError as ex:
            raise ValueError(f"arturo: source map s3://{s3_store.bucket()}/{key} is not valid JSON: {ex}")
        if raw is None:
            raise FileNotFoundError(f"arturo: no source map for {client_id} (s3://{s3_store.bucket()}/{key}); "
                                    "it lists the SharePoint roots Arturo reads, so nothing can be read without it")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"arturo: source map {key}: must be a non-empty object keyed by root")
    for root, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"source map {key}: entry for {root} must be an object")
        unknown = set(entry) - SOURCE_MAP_KEYS
        if unknown:
            raise ValueError(f"source map {key}: unknown keys {sorted(unknown)} on {root}")
        if entry.get("provider") not in PROVIDERS:
            raise ValueError(f"source map {key}: provider must be one of {sorted(PROVIDERS)} on {root}")
        if "instructions" in entry and not isinstance(entry["instructions"], str):
            raise ValueError(f"source map {key}: instructions must be a string on {root}")
        if "drop_columns" in entry and not (isinstance(entry["drop_columns"], list)
                                            and all(isinstance(c, str) for c in entry["drop_columns"])):
            raise ValueError(f"source map {key}: drop_columns must be a list of strings on {root}")
        if "source_kind" in entry and not isinstance(entry["source_kind"], str):
            raise ValueError(f"source map {key}: source_kind must be a string on {root}")
        if "skip_folders" in entry and not (isinstance(entry["skip_folders"], list)
                                            and all(isinstance(c, str) for c in entry["skip_folders"])):
            raise ValueError(f"source map {key}: skip_folders must be a list of strings on {root}")
    sharepoint_read.set_roots({root: entry["provider"] for root, entry in raw.items()})
    return raw, {"key": key, "loaded": True, "roots": list(raw)}


def _map_entry(source_map: dict, folder: str) -> dict:
    """The map entry for a payload folder: exact match, else the longest mapped root it sits
    under. Empty dict when nothing applies."""
    if folder in source_map:
        return source_map[folder]
    norm = folder.rstrip("/")
    best = ""
    for root in source_map:
        base = root.rstrip("/")
        if norm.startswith(base + "/") and len(base) > len(best):
            best = root
    return source_map.get(best, {})

# Methodology v2_3 sections 1.2 and 1.5 to 1.9, read 9/19. Weights D1..D5 = 25/20/25/15/15.
DOMAINS = {
    "D1": {"name": "Human Alignment & Trust", "weight": 0.25,
           "subs": {"D1a": "Decision Latency", "D1b": "Leadership Consistency", "D1c": "Accountability Clarity",
                    "D1d": "Founder / Key-Person Dependency", "D1e": "Cross-Functional Trust Index"}},
    "D2": {"name": "Strategic Clarity & Prioritization", "weight": 0.20,
           "subs": {"D2a": "Priority Cascade Quality", "D2b": "Resource-Strategy Alignment", "D2c": "Goal Measurability",
                    "D2d": "Strategic Plan Currency", "D2e": "Decision Framework Quality"}},
    "D3": {"name": "Operational Maturity & Process Health", "weight": 0.25,
           "subs": {"D3a": "SOP Coverage Rate", "D3b": "Handoff Effectiveness", "D3c": "KPI Measurement Maturity",
                    "D3d": "Process Stability", "D3e": "Onboarding & Knowledge Transfer"}},
    "D4": {"name": "Technology & Data Readiness", "weight": 0.15,
           "subs": {"D4a": "Tech Stack Integration Health", "D4b": "Data Quality & Accessibility", "D4c": "Reporting Infrastructure",
                    "D4d": "Tech Stack Hygiene", "D4e": "Data Room Readiness"}},
    "D5": {"name": "AI & Automation Readiness", "weight": 0.15,
           "subs": {"D5a": "Workflow Automation Readiness", "D5b": "Data Structure for ML", "D5c": "Team Change Readiness",
                    "D5d": "Integration Feasibility", "D5e": "Current Automation Maturity"}},
}

# Methodology v2_3 section 2.2, People Friction Map categories.
FRICTION = {"F1": "Leadership Communication Gaps", "F2": "Role & Seat Misalignment",
            "F3": "Cross-Functional Trust Deficits", "F4": "Meeting Culture & Decision Latency",
            "F5": "Accountability Confusion", "F6": "Change Resistance Hotspots"}

# Output schema. Enforced by the API as a tool spec (Strands structured output),
# then checked again in _validate_rows. Codes are Literal so a bad code cannot come back.
DomainCode = Literal["D1", "D2", "D3", "D4", "D5"]
SubCode = Literal["D1a", "D1b", "D1c", "D1d", "D1e", "D2a", "D2b", "D2c", "D2d", "D2e",
                  "D3a", "D3b", "D3c", "D3d", "D3e", "D4a", "D4b", "D4c", "D4d", "D4e",
                  "D5a", "D5b", "D5c", "D5d", "D5e"]
EvidenceClass = Literal["A", "B", "C", "D", "E", "F"]
ObjectType = Literal["Decision Event", "Friction Signal", "Readiness Blocker", "Automation Candidate"]
FrictionCode = Literal["F1", "F2", "F3", "F4", "F5", "F6"]


class EvidenceItem(BaseModel):
    """One evidence item extracted from one source file."""
    source_file: str = Field(description="File name exactly as given in the FILE header")
    speaker: Optional[str] = Field(description="Speaker name for transcripts, else null")
    text: str = Field(description="Close quote of the source, 40 words or fewer")
    object_type: ObjectType
    domain: DomainCode
    sub_dimension: SubCode = Field(description="Must start with the domain code")
    friction_category: Optional[FrictionCode] = Field(description="F1..F6 for a Friction Signal when one fits, else null")
    evidence_class: EvidenceClass
    half_asked: bool
    next_question: Optional[str] = Field(description="Follow-up for the next 1:1, 30 words or fewer; null unless half_asked and this is the first item raising the gap")
    live_system_hint: Optional[str] = Field(description="Canonical label of two to four words for a first-live-system candidate, else null")
    contradicts: Optional[str] = Field(description="text of an earlier item this one contradicts, else null")
    note: Optional[str] = Field(description="40 words or fewer; no client name")


class FileExtraction(BaseModel):
    """Every evidence item found in the one source file (or file chunk) in the message."""
    items: list[EvidenceItem]


class Contradiction(BaseModel):
    """Two rows from different files that cannot both be true as stated."""
    row_a: int = Field(description="Index of the first row as given in the message")
    row_b: int = Field(description="Index of the second row as given in the message")
    why: str = Field(description="30 words or fewer; what conflicts")


class ContradictionFindings(BaseModel):
    findings: list[Contradiction]


# Methodology v2_3 section 1: evidence classes and weights.
EVIDENCE_WEIGHT = {"A": 1.00, "B": 0.90, "C": 0.85, "D": 0.75, "E": 0.50, "F": 0.25}

# Per-file extraction. A file longer than MAX_CHARS_PER_CALL is split at paragraph
# boundaries; every chunk carries the same source_file. Roughly 30k tokens per call.
MAX_CHARS_PER_CALL = 120_000
CROSS_FILE_CONTRADICTIONS = os.environ.get("JB_ARTURO_CROSS_FILE", "1") == "1"
CONTRADICTION_ROW_CAP = 600      # rows sent to the contradiction call; more than this is split

# Methodology minimums. Chris 9/19: the gate uses these, nothing else.
MIN_PER_SUB = 2
MIN_PER_DOMAIN = 10
MAX_QUESTIONS_PER_SUB = 2   # backstop; the prompt asks for one question per gap


def _validate(inp: dict) -> None:
    missing = [k for k in REQUIRED_INPUT if k not in inp]
    if missing:
        raise ValueError(f"arturo input missing keys: {missing}")
    if "folders" in inp:
        raise ValueError("arturo input 'folders' is not accepted: Arturo reads the roots in "
                         "profile/<client_id>/source_map.json")
    if inp["methodology_version"] != METHODOLOGY_VERSION:
        raise ValueError(f"arturo built for methodology {METHODOLOGY_VERSION}, got {inp['methodology_version']}")
    if "graph_token" in inp:
        raise ValueError("arturo input 'graph_token' is not accepted: the runtime uses AgentCore Identity; "
                         "laptop runs set GRAPH_TOKEN in the environment")


def _short(path: str) -> str:
    """A path without the drive or site id, for people to read."""
    return path.split(":", 2)[-1] if path.startswith("list:") else path.split(":", 1)[-1]


def _list_roots(source_map: dict, since: str | None) -> tuple[list[dict], list[str]]:
    """List every file under every root (no reads). Returns (items, skipped_folders). Each item
    carries its root, its map entry and a verdict: read, transcribe, skip: older than since,
    skip: no reader for .ext."""
    items, skipped_folders = [], []
    for root in source_map:
        entry = _map_entry(source_map, root)
        for it in list_folder(root, drop_columns=entry.get("drop_columns"),
                              skip_folders=entry.get("skip_folders"), skipped=skipped_folders):
            kind = "text" if it.get("kind") == "list" else sharepoint_read.reader_for(it["ext"])
            if since and it["last_modified"] < since:
                verdict = "skip: older than since"
            elif kind == "text":
                verdict = "read"
            elif kind == "image":
                verdict = "transcribe"
            else:
                verdict = f"skip: no reader for {it['ext'] or 'files without an extension'}"
            items.append({**it, "_root": root, "_map": entry, "_verdict": verdict})
    return items, skipped_folders


def _plan_rows(items: list[dict]) -> list[dict]:
    return [{"verdict": it["_verdict"], "root": _short(it["_root"]), "path": _short(it["path"]),
             "type": it["ext"], "size": it.get("size", 0), "modified": it.get("last_modified", "")}
            for it in items]


IMAGE_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".webp": "webp"}
TRANSCRIBE_PROMPT = ("Transcribe every piece of text in this image exactly as written, including every "
                     "table cell. Write each table row on one line with cells separated by ' | '. Keep "
                     "numbers, names and titles exactly. Output only the transcription, no commentary.")


def _transcribe_bytes(fmt: str, data: bytes) -> str:
    """One model call: one picture in, plain text out."""
    import boto3
    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=BotocoreConfig(retries={"max_attempts": 2, "mode": "standard"}, read_timeout=300))
    r = client.converse(modelId=MODEL_ID, inferenceConfig={"maxTokens": 8000, "temperature": 0},
                        messages=[{"role": "user", "content": [
                            {"image": {"format": fmt, "source": {"bytes": data}}},
                            {"text": TRANSCRIBE_PROMPT}]}])
    return "".join(c.get("text", "") for c in r["output"]["message"]["content"]).strip()


def _transcribe_image(item: dict) -> str:
    """An image file: one model call. The text is then extracted like any file."""
    text = _transcribe_bytes(IMAGE_FORMATS[item["ext"]], sharepoint_read.read_file_bytes(item))
    if not text:
        raise ValueError(f"image transcription returned no text: {item['name']}")
    return f"[Transcribed from an image by the model; check figures against the original.]\n\n{text}"


def _read_via_images(item: dict) -> str:
    """A pdf/docx/odt/pptx with no text: transcribe the pictures inside it (10/12, Lindsey's
    exported flowcharts and scanned PDFs). Raises when there are none the model can take."""
    imgs, unsupported = sharepoint_read.embedded_images(item)
    if not imgs:
        why = "no text and no pictures the model can read"
        if unsupported:
            why += f" (picture formats not supported: {', '.join(unsupported)})"
        raise ValueError(why)
    parts = []
    for i, (fmt, data) in enumerate(imgs, 1):
        t = _transcribe_bytes(fmt, data)
        if t:
            parts.append(f"--- Picture {i} ---\n{t}")
    if not parts:
        raise ValueError("pictures found but transcription returned no text")
    note = (f"[No typed text in this file. Transcribed by the model from {len(imgs)} picture(s) inside it; "
            "check figures against the original.]")
    if len(imgs) >= sharepoint_read.MAX_IMAGES_PER_FILE:
        note += f" [Only the first {sharepoint_read.MAX_IMAGES_PER_FILE} pictures were read.]"
    return note + "\n\n" + "\n\n".join(parts)


def _read_one(it: dict) -> tuple[str, bool]:
    """Text for one file and whether it came from pictures. Raises on failure."""
    if it["_verdict"] == "transcribe":
        return _transcribe_image(it), True
    try:
        text = read_file_text(it)
    except ValueError as e:
        if it["ext"] not in sharepoint_read.IMAGE_FALLBACK_EXT or "no text layer" not in str(e):
            raise
        text = ""
    if text.strip():
        return text, False
    if it["ext"] in sharepoint_read.IMAGE_FALLBACK_EXT:
        return _read_via_images(it), True
    raise ValueError("file has no text")


def _collect_sources(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Read every file whose verdict is read or transcribe, WORKERS at a time, results kept in
    plan order. Returns (sources, read_failures). A file that will not read is recorded with its
    error and the run continues."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [it for it in items if it["_verdict"] in ("read", "transcribe")]

    def one(it):
        try:
            text, via_pictures = _read_one(it)
            return {**it, "text": text, "_via_pictures": via_pictures}, None
        except Exception as e:  # noqa: BLE001  recorded per file, run continues
            return None, {"file": it["name"], "path": _short(it["path"]), "stage": "read",
                          "error": f"{type(e).__name__}: {e}"[:400]}

    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        results = list(pool.map(one, todo))
    sources = [s for s, _ in results if s]
    failed = [f for _, f in results if f]
    return sources, failed


OBJECT_TYPES = {"Decision Event", "Friction Signal", "Readiness Blocker", "Automation Candidate"}
ROW_KEYS = ("source_file", "speaker", "text", "object_type", "domain", "sub_dimension",
            "friction_category", "evidence_class", "half_asked", "next_question", "live_system_hint", "contradicts", "note")
# Set in code after validation, never by the model: source_kind from the source map (None when
# the root has no entry), run_id in work().


def _system_prompt() -> str:
    """Prompt, writing standard, methodology rules. Read from the repo, one copy each, no build step."""
    return (SYSTEM_PROMPT_PATH.read_text() + "\n\n" + WRITING_STANDARD_PATH.read_text()
            + "\n\n" + RULES_PATH.read_text())


def _wc(s) -> int:
    return len(str(s).split()) if s else 0


def _check_writing(rows: list[dict]) -> dict:
    """Length and vocabulary checks on model-authored fields. Pure code.

    Returns lists for the report. Never raises: a row over cap is evidence with a long
    quote, and the report shows it; hard_over lists the gross ones separately.
    """
    over, banned, hard = [], [], []
    caps = {"text": CAP_TEXT, "next_question": CAP_QUESTION, "note": CAP_NOTE, "live_system_hint": CAP_LABEL}
    for i, r in enumerate(rows):
        for f, cap in caps.items():
            n = _wc(r.get(f))
            if n > cap:
                over.append(f"row {i} {f}: {n} words, cap {cap}")
            if n > cap * HARD_OVER:
                hard.append(f"row {i} {f}: {n} words, cap {cap}")
        for f in AUTHORED_FIELDS:
            words = {w.strip(".,;:?!\"'()").lower() for w in str(r.get(f) or "").split()}
            hit = words & BANNED_WORDS
            if hit:
                banned.append(f"row {i} {f}: {sorted(hit)}")
            if r.get(f) and ("\u2014" in r[f] or "\u2013" in r[f]):
                banned.append(f"row {i} {f}: dash")
    # 2026-09-24: this used to raise on a gross overrun and killed the 33-file client run
    # on one 73-word quote after 498 rows were extracted. A long quote is reported, not fatal.
    return {"over_cap": over, "banned_words": banned, "hard_over": hard}


def _chunks(text: str, max_chars: int | None = None) -> list[str]:
    """Split long text at paragraph boundaries. Most files return as one chunk."""
    max_chars = MAX_CHARS_PER_CALL if max_chars is None else max_chars
    if len(text) <= max_chars:
        return [text]
    out, buf, size = [], [], 0
    for para in text.split("\n"):
        if size + len(para) + 1 > max_chars and buf:
            out.append("\n".join(buf)); buf, size = [], 0
        buf.append(para); size += len(para) + 1
    if buf:
        out.append("\n".join(buf))
    return out


def _build_user_message(source: dict, text: str, part: int, parts: int) -> str:
    head = f"One source file follows. Extract every evidence item in it."
    if parts > 1:
        head += f" This is part {part} of {parts} of the same file; items from other parts are extracted separately."
    # Name on its own line. The model copied "name (modified ...)" into source_file on 10 of 33
    # client files; source_file is set in code now, but keep the header unambiguous.
    msg = f"{head}\n\n===== FILE: {source['name']} =====\nModified: {source['last_modified']}\n\n{text}"
    instructions = (source.get("_map") or {}).get("instructions", "").strip()
    if instructions:
        # Source map: per-root guidance for this file only. Placed after the text so the
        # model reads the source first, then what the source is. Every other file is unchanged.
        msg += f"\n\n===== INSTRUCTIONS FOR THIS FILE (from the client source map) =====\n{instructions}"
    return msg


def _model() -> BedrockModel:
    """One client for the run. read_timeout stays long because structured output arrives
    as a single tool call and the stream can be quiet; retries are 2, not 3, because a
    retry inside the runtime doubles the model spend with no signal to the caller."""
    boto_config = BotocoreConfig(
        retries={"max_attempts": 2, "mode": "standard"},
        connect_timeout=10,
        read_timeout=900,
    )
    return BedrockModel(
        model_id=MODEL_ID,
        region_name=REGION,
        max_tokens=32000,
        temperature=0,
        cache_config=CacheConfig(strategy="auto"),
        boto_client_config=boto_config,
    )


USAGE_KEYS = ("inputTokens", "outputTokens", "totalTokens", "cacheReadInputTokens", "cacheWriteInputTokens")


def _add_usage(total: dict, u: dict) -> None:
    for k in USAGE_KEYS:
        total[k] = total.get(k, 0) + int(u.get(k, 0) or 0)


def _extract_one(model: BedrockModel, system_prompt: str, source: dict, text: str,
                 part: int, parts: int) -> tuple[list[dict], dict]:
    """One model call, one file (or chunk). Strands structured output: the Pydantic model
    becomes a Bedrock tool spec, the model must call it, the reply is schema-checked before
    we see it. Rows are then checked again against the methodology codes."""
    agent = Agent(model=model, system_prompt=system_prompt, tools=[], callback_handler=None)
    result = agent(_build_user_message(source, text, part, parts), structured_output_model=FileExtraction)
    usage = dict(result.metrics.accumulated_usage)
    parsed = result.structured_output
    if parsed is None:
        path = _save_raw(str(result), f"no_structured_output_{_safe(source['name'])}")
        raise ValueError(f"model returned no structured output; raw reply saved to {path}")
    items = [it.model_dump() for it in parsed.items]
    # One file per call, so the source is known. Set it in code; do not ask the model to
    # reproduce a file name. Count the mismatches so the prompt can be judged later.
    mismatched = sum(1 for it in items if it.get("source_file") != source["name"])
    for it in items:
        it["source_file"] = source["name"]
    try:
        rows = _validate_rows(items, [source])
    except ValueError:
        _save_raw(parsed.model_dump_json(indent=2), f"validation_failed_{_safe(source['name'])}")
        raise
    kind = (source.get("_map") or {}).get("source_kind")
    for r in rows:
        r["source_kind"] = kind
    usage["source_file_mismatches"] = mismatched
    return rows, usage


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:60]


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _merge_ledger(rows: list[dict]) -> tuple[list[dict], int]:
    """Deterministic. Drop exact repeats (same file, same sub-dimension, same text after
    whitespace and case normalisation). The same fact from two different files is kept:
    that is corroboration, counted in coverage, not a duplicate."""
    seen, out, dropped = set(), [], 0
    for r in rows:
        key = (r["source_file"], r["sub_dimension"], _norm(r["text"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


WORKERS = int(os.environ.get("JB_ARTURO_WORKERS", "4"))


def _extract_file(model: BedrockModel, system_prompt: str, s: dict) -> tuple[list[dict], dict, int]:
    """All chunks of one file, serially. Returns (rows, usage, chunk_count)."""
    chunks = _chunks(s["text"])
    rows, usage = [], {}
    for j, chunk in enumerate(chunks, 1):
        r, u = _extract_one(model, system_prompt, s, chunk, j, len(chunks))
        rows.extend(r)
        _add_usage(usage, u)
    return rows, usage, len(chunks)


def _extract_evidence(sources: list[dict], progress=None,
                      prior_failed: list[dict] | None = None) -> tuple[list[dict], dict, list[dict]]:
    """Model step, one call per file or chunk, WORKERS files at a time. Returns (rows, usage, files_failed).

    Files run in a thread pool; results are placed by file index so the ledger is the
    same as a serial run. A failing file is recorded with its error (visible live in
    status.json) and the run continues. Chris rule: a gap is reported, never papered
    over, and one bad transcript must not cost the other 32.
    """
    if not sources:
        raise ValueError("arturo: no source files found in the given folders")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    model = _model()
    system_prompt = _system_prompt()
    results: list = [None] * len(sources)
    failed: list[dict] = []
    prior = list(prior_failed or [])     # read failures, shown in status alongside extraction failures
    per_file: list = [None] * len(sources)
    usage: dict = {"model_calls": 0, "workers": WORKERS}
    done = 0
    lock = __import__("threading").Lock()

    def submit_all(pool):
        return {pool.submit(_extract_file, model, system_prompt, s): i for i, s in enumerate(sources)}

    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        futures = submit_all(pool)
        for fut in as_completed(futures):
            i = futures[fut]
            s = sources[i]
            try:
                rows, u, n_chunks = fut.result()
                with lock:
                    results[i] = rows
                    per_file[i] = {"file": s["name"], "chunks": n_chunks, "rows": len(rows), **u}
                    usage["model_calls"] += n_chunks
                    _add_usage(usage, u)
            except Exception as e:  # noqa: BLE001  recorded per file, run continues
                with lock:
                    failed.append({"file": s["name"], "error": f"{type(e).__name__}: {e}"[:400]})
                    per_file[i] = {"file": s["name"], "rows": 0, "failed": True}
            with lock:
                done += 1
                if progress:
                    progress({"files_done": done, "files_failed": prior + list(failed),
                              "rows_so_far": sum(len(r) for r in results if r)})

    rows_all = [r for rows in results if rows for r in rows]
    usage["per_file"] = per_file
    if not rows_all and failed:
        raise ValueError("arturo: every source file failed extraction; first error: " + failed[0]["error"])
    merged, dropped = _merge_ledger(rows_all)
    usage["duplicates_dropped"] = dropped
    usage["writing_check"] = _check_writing(merged)
    return merged, usage, failed


def _cross_file_contradictions(ledger: list[dict], usage: dict) -> list[dict]:
    """Second model call, small: compact rows in, pairs out. Within-file contradictions
    already arrive on the row's own 'contradicts' field; this looks across files."""
    if not CROSS_FILE_CONTRADICTIONS or len(ledger) < 2:
        return []
    model = _model()
    prompt = ("You read evidence rows from a Clarity Sprint. Each row has an index, a sub-dimension, "
              "a source file, a speaker and a short quote. Report only pairs from DIFFERENT source files "
              "that cannot both be true as stated. Same topic with different emphasis is not a "
              "contradiction. Return no finding rather than a doubtful one.")
    findings: list[dict] = []
    for start in range(0, len(ledger), CONTRADICTION_ROW_CAP):
        batch = ledger[start:start + CONTRADICTION_ROW_CAP]
        lines = [f"[{start + i}] {r['sub_dimension']} | {r['source_file']} | {r.get('speaker') or '-'} | {r['text']}"
                 for i, r in enumerate(batch)]
        agent = Agent(model=model, system_prompt=prompt, tools=[], callback_handler=None)
        result = agent("\n".join(lines), structured_output_model=ContradictionFindings)
        _add_usage(usage, dict(result.metrics.accumulated_usage))
        usage["model_calls"] = usage.get("model_calls", 0) + 1
        parsed = result.structured_output
        if parsed is None:
            continue
        for f in parsed.findings:
            if not (0 <= f.row_a < len(ledger) and 0 <= f.row_b < len(ledger)) or f.row_a == f.row_b:
                continue
            a, b = ledger[f.row_a], ledger[f.row_b]
            if a["source_file"] == b["source_file"]:
                continue
            findings.append({"later": b["text"], "earlier": a["text"], "file": b["source_file"],
                             "earlier_file": a["source_file"], "why": f.why, "scope": "cross-file"})
    return findings


def _writable_dir(preferred: str) -> str | None:
    """Return a directory we can write to: the preferred one, else /tmp/<name>, else None.

    The AgentCore runtime mounts the code read-only, so a relative 'output' folder fails
    there; /tmp is writable. Locally the preferred path works.
    """
    for candidate in (preferred, os.path.join("/tmp", os.path.basename(preferred.rstrip("/")) or "output")):
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return candidate
        except OSError:
            continue
    return None


def _save_raw(text: str, label: str) -> str | None:
    """Keep the model's raw reply whenever a run fails, so the failure can be read later."""
    d = _writable_dir("output")
    if d is None:
        return None
    path = os.path.join(d, f"arturo_raw_{label}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.txt")
    with open(path, "w") as f:
        f.write(text)
    return path


def _validate_rows(items: list, sources: list[dict]) -> list[dict]:
    """Reject anything outside the methodology codes. A bad row is an error, not a fix."""
    valid_subs = {sub for d in DOMAINS.values() for sub in d["subs"]}
    names = {s["name"] for s in sources}
    rows, problems = [], []
    for i, it in enumerate(items):
        missing = [k for k in ROW_KEYS if k not in it]
        if missing:
            problems.append(f"item {i}: missing {missing}"); continue
        if it["domain"] not in DOMAINS:
            problems.append(f"item {i}: bad domain {it['domain']!r}"); continue
        if it["sub_dimension"] not in valid_subs or not it["sub_dimension"].startswith(it["domain"]):
            problems.append(f"item {i}: bad sub_dimension {it['sub_dimension']!r}"); continue
        if it["friction_category"] is not None and it["friction_category"] not in FRICTION:
            problems.append(f"item {i}: bad friction_category {it['friction_category']!r}"); continue
        if it["evidence_class"] not in EVIDENCE_WEIGHT:
            problems.append(f"item {i}: bad evidence_class {it['evidence_class']!r}"); continue
        if it["object_type"] not in OBJECT_TYPES:
            problems.append(f"item {i}: bad object_type {it['object_type']!r}"); continue
        if it["source_file"] not in names:
            problems.append(f"item {i}: unknown source_file {it['source_file']!r}"); continue
        it["half_asked"] = bool(it["half_asked"])
        rows.append({k: it[k] for k in ROW_KEYS})
    if problems:
        raise ValueError("arturo: model output failed validation:\n" + "\n".join(problems))
    return rows


def _counted_rows(rows: list[dict]) -> list[dict]:
    """Rows that count toward coverage for one sub-dimension. Pure code.

    A row is full when the source answered the point. A half-asked row (fact stated,
    follow-up not pursued in that file) counts when a DIFFERENT file holds a full row
    on the same sub-dimension: the corpus answered it even though this file did not.
    A sub-dimension with only half-asked rows still counts nothing. Added 2026-09-24
    after per-file extraction tripled half-asked rows (11 to 32 on one test set) because the
    model no longer sees the other files; 27 of the 32 had a full row elsewhere.
    The half_asked flag and next_question stay on the row for the questions list."""
    full = [r for r in rows if not r.get("half_asked")]
    full_files = {r["source_file"] for r in full}
    answered = [r for r in rows if r.get("half_asked") and (full_files - {r["source_file"]})]
    return full + answered


def _coverage_and_gate(ledger: list[dict]) -> tuple[dict, dict]:
    """Pure arithmetic. No model."""
    coverage, failing = {}, []
    for d, spec in DOMAINS.items():
        coverage[d] = {"name": spec["name"], "weight": spec["weight"], "subs": {}, "items": 0}
        for sub in spec["subs"]:
            rows = [r for r in ledger if r["domain"] == d and r["sub_dimension"] == sub]
            full = _counted_rows(rows)
            weights = [EVIDENCE_WEIGHT[r["evidence_class"]] for r in full]
            if len(full) >= MIN_PER_SUB:
                status = "captured"
            elif len(full) == 1:
                status = "thin"          # one real item; needs a second source
            elif rows:
                status = "half asked"    # only half-asked items, nothing complete
            else:
                status = "missing"
            provisional = bool(full) and all(r["evidence_class"] in ("E", "F") for r in full)
            coverage[d]["subs"][sub] = {
                "name": spec["subs"][sub],
                "status": status,
                "provisional": provisional,
                "items": len(full),
                "sources": len({r["source_file"] for r in full}),
                "half_asked": sum(1 for r in rows if r.get("half_asked")),
                "half_asked_open": len(rows) - len(full),
                "avg_weight": round(sum(weights) / len(weights), 2) if weights else 0.0,
            }
            coverage[d]["items"] += len(full)
            if len(full) < MIN_PER_SUB:
                failing.append(f"{d} {sub}: {len(full)} of {MIN_PER_SUB}")
        if coverage[d]["items"] < MIN_PER_DOMAIN:
            failing.append(f"{d}: {coverage[d]['items']} of {MIN_PER_DOMAIN}")
        dom_weights = [EVIDENCE_WEIGHT[r["evidence_class"]] for r in _counted_rows([r for r in ledger if r["domain"] == d])]
        coverage[d]["confidence"] = round(sum(dom_weights) / len(dom_weights), 2) if dom_weights else 0.0
    return coverage, {"passed": not failing, "failing": failing}


def _next_best_questions(ledger: list[dict], coverage: dict) -> list[dict]:
    """Questions for the next 1:1. Ordered by domain weight.

    Half-asked items carry the model's follow-up question, and every sub-dimension that
    has one lists it, captured or not. A full row in another file counts toward coverage
    but does not show this follow-up was asked. 2026-09-29: captured sub-dimensions used
    to be skipped, so any run that passed the gate returned no questions (several runs).
    Sub-dimensions that are not captured and have no follow-up carry a placeholder; the
    person running the 1:1 writes one from the code.
    """
    out = []
    for d, spec in sorted(coverage.items(), key=lambda kv: -kv[1]["weight"]):
        for sub, c in spec["subs"].items():
            qs = list(dict.fromkeys((r.get("next_question") or "").strip() for r in ledger
                      if r["sub_dimension"] == sub and r.get("half_asked")
                      and (r.get("next_question") or "").strip()))
            if qs:
                for q in qs[:MAX_QUESTIONS_PER_SUB]:
                    out.append({"domain": d, "sub_dimension": sub, "status": c["status"], "question": q})
            elif c["status"] == "captured":
                continue
            elif c["status"] == "thin":
                out.append({"domain": d, "sub_dimension": sub, "status": c["status"],
                            "question": f"Second source needed for {c['name']}: one item so far."})
            else:
                out.append({"domain": d, "sub_dimension": sub, "status": c["status"],
                            "question": f"Not yet asked: {c['name']}."})
    return out


def _write_outputs(inp: dict, result: dict) -> dict | None:
    """Best effort. The report is also returned inline as result['markdown']."""
    d = _writable_dir(inp["output_dir"])
    if d is None:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = os.path.join(d, f"arturo_{inp['client_id']}_{inp['run_id']}_{stamp}")
    with open(base + ".json", "w") as f:
        json.dump(result, f, indent=2)
    with open(base + ".md", "w") as f:
        f.write(result["markdown"])
    return {"json": base + ".json", "md": base + ".md"}


def _render_md(inp: dict, r: dict) -> str:
    sm = r.get("source_map", {})
    sm_line = f"Source map: {sm.get('key')}, roots read: {len(sm.get('roots', []))}."
    for root, d in sorted(sm.get("dropped_columns", {}).items()):
        sm_line += f" Columns left out of {root.split(':')[-1]}: {', '.join(d)}."
    lines = [f"# Arturo S. evidence report: {inp['client_id']} run {inp['run_id']}", "",
             "Produced with AI support. Requires human review before use.", "",
             f"Signed DFM: {r.get('dfm_profile', 'not recorded')}.", "",
             f"Gate: {'PASSED' if r['gate']['passed'] else 'NOT PASSED'}", "",
             f"Files read: {len(r['sources_read'])}. Files failed: {len(r.get('files_failed', []))}. "
             f"Ledger rows: {len(r['ledger'])}.", sm_line, ""]
    if r.get("files_failed"):
        lines.append("## Files that did not extract (not in the ledger; run again after the cause is fixed)")
        for f in r["files_failed"]:
            lines.append(f"- {f.get('path', f['file'])}: {f['error']}")
        lines.append("")
    if sm.get("skipped_folders") or sm.get("skipped_files"):
        lines.append("## Skipped by rule (not read)")
        for f in sm.get("skipped_folders", []):
            lines.append(f"- folder {f}: skip-listed folder")
        for f in sm.get("skipped_files", []):
            lines.append(f"- {f['path']}: {f['why']}")
        lines.append("")
    for d, c in r["coverage"].items():
        lines.append(f"## {d} {c['name']} (items {c['items']}, confidence {c['confidence']})")
        for sub, s in c["subs"].items():
            flag = " PROVISIONAL" if s["provisional"] else ""
            lines.append(f"- {sub} {s['name']}: {s['status']}{flag} ({s['items']} items from {s['sources']} sources, {s['half_asked']} half asked of which {s['half_asked_open']} open, avg weight {s['avg_weight']})")
        lines.append("")
    lines.append("## Next-best questions for the next 1:1")
    for q in r["next_best_questions"]:
        lines.append(f"- [{q['domain']} {q['sub_dimension']}, {q['status']}] {q['question'] or 'TODO'}")
    lines.append("")
    wc = r.get("writing_check", {})
    lines.append("## Writing check")
    for key, label in (("over_cap", "fields over cap"),
                       ("banned_words", "banned words or dashes in authored fields"),
                       ("hard_over", "quotes grossly over cap (kept, flagged)")):
        items = wc.get(key, [])
        lines.append(f"- {label}: {len(items)}")
        for item in items[:10]:
            lines.append(f"  - {item}")
        if len(items) > 10:
            lines.append(f"  - and {len(items) - 10} more in the JSON result")
    lines.append("")
    lines.append("## Contradictions to reconcile")
    if not r["contradictions"]:
        lines.append("- none found")
    for c in r["contradictions"]:
        if c.get("scope") == "cross-file":
            lines.append(f"- {c['file']}: \"{c['later']}\" vs {c['earlier_file']}: \"{c['earlier']}\" ({c['why']})")
        else:
            lines.append(f"- {c['file']}: \"{c['later']}\" vs earlier \"{c['earlier']}\"")
    lines.append("")
    lines.append("## First live system signal for Fe D.")
    for s in r["live_system_signal"]:
        lines.append(f"- {s}")
    return "\n".join(lines) + "\n"


def _set_identity() -> None:
    if graph_identity.on_runtime():
        # AgentCore Runtime: Arturo's own identity for Slate Seven folders ("s7"), and the
        # client-tenant app registration for roots the source map marks provider "client". The second provider is set only
        # when the target's CDK passed JB_CLIENT_GRAPH_PROVIDER; a client root with no
        # provider fails at token time, not silently. Laptop runs keep sharepoint_read's
        # defaults (GRAPH_TOKEN, GRAPH_TOKEN_CLIENT).
        sharepoint_read.set_token_provider(graph_identity.cached_provider(), "s7")
        if graph_identity.CLIENT_PROVIDER_NAME:
            sharepoint_read.set_token_provider(
                graph_identity.cached_provider(graph_identity.CLIENT_PROVIDER_NAME), "client")


def prepare(inp: dict) -> dict:
    """Foreground half (seconds). Validates and lists every root so a bad root or a missing
    grant fails the caller immediately. Returns the plan: every file found, with its verdict,
    so the start response shows exactly what the run reads and skips."""
    _validate(inp)
    dfm = _require_dfm(inp)
    _set_identity()
    source_map, _ = _load_source_map(inp)
    items, skipped_folders = _list_roots(source_map, inp.get("since"))
    to_read = [it for it in items if it["_verdict"] in ("read", "transcribe")]
    if not to_read:
        raise ValueError("arturo: no readable source files found under the source map's roots")
    counts: dict = {}
    for it in items:
        v = it["_verdict"] if not it["_verdict"].startswith("skip: no reader") else "skip: no reader"
        counts[v] = counts.get(v, 0) + 1
    return {"files_total": len(to_read), "files_done": 0, "files_failed": [],
            "files": [it["name"] for it in to_read],
            "plan": _plan_rows(items), "plan_counts": counts,
            "skipped_folders": [_short(f) for f in skipped_folders],
            "roots": [_short(r) for r in source_map], "dfm_profile": dfm}


def work(inp: dict, progress=None) -> dict:
    """Background half. Reads, extracts per file, merges, gates, renders."""
    dfm = _require_dfm(inp)
    _set_identity()
    source_map, map_info = _load_source_map(inp)
    items, skipped_folders = _list_roots(source_map, inp.get("since"))
    if progress:
        progress({"phase": "reading", "plan": _plan_rows(items)})
    sources, read_failed = _collect_sources(items)
    map_info["skipped_folders"] = [_short(f) for f in skipped_folders]
    map_info["skipped_files"] = [{"path": _short(it["path"]), "why": it["_verdict"]}
                                 for it in items if it["_verdict"] not in ("read", "transcribe")]
    map_info["transcribed_images"] = [s["name"] for s in sources if s.get("_via_pictures")]
    map_info["dropped_columns"] = {s["path"]: s["dropped_columns"] for s in sources if s.get("dropped_columns")}
    map_info["files_with_instructions"] = [s["name"] for s in sources if (s.get("_map") or {}).get("instructions")]
    if progress:
        progress({"files_total": len(sources), "phase": "extracting", "source_map": map_info,
                  "files_failed": list(read_failed)})
    ledger, usage, failed = _extract_evidence(sources, progress, read_failed)
    failed = read_failed + failed
    for row in ledger:
        row["run_id"] = inp["run_id"]
    if progress:
        # The ledger is the expensive part. Park it in status now so a failure in the
        # cheap steps below (gate, render) cannot lose it.
        progress({"phase": "merged", "partial": {"ledger": ledger, "files_failed": failed, "usage": usage}})
    coverage, gate = _coverage_and_gate(ledger)
    if progress:
        progress({"phase": "contradictions"})
    within = [{"later": r["text"], "earlier": r["contradicts"], "file": r["source_file"], "scope": "within-file"}
              for r in ledger if r.get("contradicts")]
    result = {
        "ledger": ledger,
        "coverage": coverage,
        "gate": gate,
        "next_best_questions": _next_best_questions(ledger, coverage),
        "live_system_signal": sorted({r["live_system_hint"].strip().lower() for r in ledger if r.get("live_system_hint")}),
        "contradictions": within + _cross_file_contradictions(ledger, usage),
        "sources_read": [s["path"] for s in sources],
        "files_failed": failed,
        "source_map": map_info,
        "dfm_profile": dfm,
    }
    result["writing_check"] = usage.pop("writing_check", {})
    result["markdown"] = _render_md(inp, result)
    result["files"] = _write_outputs(inp, result)
    if progress:
        progress({"phase": "complete"})
    return {"output": result, "usage": usage}


def run(inp: dict) -> dict:
    """Synchronous: laptop runs, tests, and test payloads. Same result as start + status."""
    prepare(inp)
    return work(inp)
