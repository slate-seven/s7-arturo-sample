"""Read-only SharePoint access over Microsoft Graph.

Three operations, nothing else: list a folder, read a file, read a list. No write,
no delete, no move. Damon's rule: federated credential via AgentCore Identity,
certificate fallback, no client secret, outbound only, logged.

Allowlist is enforced here as well as by Sites.Selected permission. The caller sets it per
run with set_roots() from the client's source map (2026-10-11, portability review R01 to R03,
R06): no client root lives in this file. A root outside it raises before any network call,
and an unset allowlist means every call fails.

Token acquisition is injected per provider key. Two keys exist: "s7" (the Slate
Seven app registration, our tenant) and "client" (an app registration in the
client's own tenant). Each root carries its key in the allowlist. On a laptop the defaults read
GRAPH_TOKEN and GRAPH_TOKEN_CLIENT from the environment. On AgentCore Runtime
arturo.py registers graph_identity providers for both keys.

Root formats (one string each):
    "<drive-id>:<path/under/root>"           a document-library folder
        e.g. "b!<drive_id>:Folder/Subfolder"
    "list:<site-id>:<list display name>"     a SharePoint list, read as one document
        e.g. "list:contoso.sharepoint.com,1111...,2222...:AI Roster"
Drive ids and site ids come from the client's access plan, through the source map.

A list root yields exactly one item from list_folder(). read_file_text() on that
item returns a Markdown table: one row per list item, one column per user-defined
list column, person lookups reduced to display name (no email), system columns
dropped. The item's last_modified is the newest row's modification time so the
"since" filter works the same way it does for files.

drop_columns (2026-10-09): list_folder() takes an optional list of column names to leave
out of the rendered table, matched on display name or internal name, case-insensitive.
Applied in code before the text exists, so a dropped column never reaches a model. The
caller (arturo.py) takes the list from the client's source map in S3; the default is to
drop nothing. Folder roots ignore it.

2026-10-12 (client source list):
- A root may be a single file ("<drive-id>:<path/to/file.docx>"), not only a folder. It is found
  by listing its parent folder; only folder listings are used, never a lookup of the item.
- Folders named in SKIP_FOLDERS (Archive, Agent Pack Outputs) are never walked, at any depth,
  on any root; a root can name more in the source map (skip_folders). Skipped folders are
  reported to the caller, never silently dropped.
- Readers for pptx, xlsx, pdf, odt and html join txt/md/csv/json/docx. Images come back as
  bytes; arturo.py transcribes them. Anything else is reported as "no reader", not read.
- Every reader raises on failure; the caller records the file as failed and keeps going.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import quote

import requests

GRAPH = "https://graph.microsoft.com/v1.0"
TIMEOUT = 60

# Readable roots for this run, root -> provider key ("s7" or "client"). Set by the caller from
# the client's source map. Empty means every call fails.
ALLOWED_ROOTS: dict[str, str] = {}

TEXT_EXT = {".txt", ".md", ".csv", ".json"}
DOCX_EXT = {".docx"}
PPTX_EXT = {".pptx"}
XLSX_EXT = {".xlsx", ".xlsm"}
PDF_EXT = {".pdf"}
ODT_EXT = {".odt"}
HTML_EXT = {".html", ".htm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}   # read as bytes; arturo.py transcribes
READABLE_EXT = TEXT_EXT | DOCX_EXT | PPTX_EXT | XLSX_EXT | PDF_EXT | ODT_EXT | HTML_EXT

# Folder names never walked, matched case-insensitively at any depth (Katie 10/12: ignore
# archive folders on both sides; Agent Pack Outputs holds the agents' own outputs).
SKIP_FOLDERS = {"archive", "agent pack outputs"}
XLSX_MAX_ROWS = 3000        # per sheet; a sheet longer than this is cut and says so

# Image fallback (10/12): a pdf/docx/odt/pptx with no text often holds its content as pictures
# (exported flowcharts, scans). embedded_images() pulls them out for arturo.py to transcribe.
IMAGE_FALLBACK_EXT = {".pdf", ".docx", ".odt", ".pptx"}
MAX_IMAGES_PER_FILE = 40
MIN_IMAGE_SIDE = 200        # px; smaller pictures are logos and icons, not content
MAX_IMAGE_BYTES = 3_500_000  # Bedrock image limit is 3.75 MB
MAX_IMAGE_SIDE = 7900        # Bedrock limit is 8000 px

# List columns Graph returns on every list that carry no client content.
_LIST_SKIP_COLUMNS = {"Attachments", "ContentType", "Title_x0020_Display"}
_LIST_ENV = {"s7": "GRAPH_TOKEN", "client": "GRAPH_TOKEN_CLIENT"}

_token_providers: dict[str, Callable[[], str]] = {}


def set_roots(roots: dict[str, str]) -> None:
    """Replace the allowlist for this run: {root: provider key}."""
    bad = {r: k for r, k in roots.items() if k not in _LIST_ENV}
    if bad:
        raise ValueError(f"unknown provider keys: {bad}")
    ALLOWED_ROOTS.clear()
    ALLOWED_ROOTS.update(roots)


def set_token_provider(fn: Callable[[], str], key: str = "s7") -> None:
    _token_providers[key] = fn


def _token(key: str = "s7") -> str:
    fn = _token_providers.get(key)
    if fn:
        return fn()
    env = _LIST_ENV.get(key, "GRAPH_TOKEN")
    tok = os.environ.get(env)
    if not tok:
        raise RuntimeError(f"no Graph token for provider {key!r}: set {env} or call set_token_provider(fn, {key!r})")
    return tok


def _headers(key: str = "s7") -> dict:
    return {"Authorization": f"Bearer {_token(key)}"}


def _is_list_root(root: str) -> bool:
    return root.startswith("list:")


def _split(folder: str) -> tuple[str, str]:
    parts = folder.split(":", 1)
    if len(parts) != 2 or not parts[0].startswith("b!"):
        raise ValueError(f"folder must be '<drive-id>:<path>', got {folder!r}")
    return parts[0], parts[1].strip("/")


def _split_list(root: str) -> tuple[str, str]:
    # "list:<site-id>:<list name>"; the site id itself contains commas, never colons.
    parts = root.split(":", 2)
    if len(parts) != 3 or parts[0] != "list" or not parts[1] or not parts[2]:
        raise ValueError(f"list root must be 'list:<site-id>:<list name>', got {root!r}")
    return parts[1], parts[2]


def _check_allowed(path: str) -> str:
    """Raise unless path is inside an allowed root. Returns the matching root."""
    if not ALLOWED_ROOTS:
        raise PermissionError("no roots set for this run; no SharePoint folder is readable")
    norm = path.rstrip("/")
    for r in ALLOWED_ROOTS:
        base = r.rstrip("/")
        if norm == base or norm.startswith(base + "/"):
            return r
    raise PermissionError(f"path not in allowlist: {path}")


def _provider_key(path: str) -> str:
    return ALLOWED_ROOTS[_check_allowed(path)]


def _file_entry(drive: str, path: str, it: dict) -> dict:
    name = it["name"]
    return {
        "path": f"{drive}:{path}",
        "name": name,
        "item_id": it["id"],
        "drive_id": drive,
        "size": it.get("size", 0),
        "last_modified": it.get("lastModifiedDateTime", ""),
        "ext": os.path.splitext(name)[1].lower(),
    }


def list_folder(folder: str, recurse: bool = True, drop_columns: list[str] | None = None,
                skip_folders: list[str] | None = None, skipped: list | None = None,
                _top: bool = True) -> list[dict]:
    """Return files under the root. Each: {path, name, item_id, drive_id, size, last_modified, ext}.
    A list root returns one synthetic item carrying the rendered list (see _list_item);
    drop_columns names list columns to leave out of it. A file root returns that one file.
    Folders named in SKIP_FOLDERS or skip_folders are not walked; their paths are appended to
    `skipped` when the caller passes a list."""
    if _is_list_root(folder):
        _check_allowed(folder)
        return [_list_item(folder, drop_columns)]
    key = _provider_key(folder)
    drive, path = _split(folder)
    skip = SKIP_FOLDERS | {f.strip().lower() for f in (skip_folders or [])}
    if _top and path and _looks_like_file(path):
        # A file root: list its parent folder and keep that one file. Uses only the folder
        # listing call; a lookup of the item itself 404s under item-level grants (10/12).
        parent, _, fname = path.rpartition("/")
        purl = (f"{GRAPH}/drives/{drive}/root:/{parent}:/children" if parent
                else f"{GRAPH}/drives/{drive}/root/children")
        while purl:
            r = requests.get(purl, headers=_headers(key), timeout=TIMEOUT)
            r.raise_for_status()
            body = r.json()
            for it in body.get("value", []):
                if it["name"] == fname and "file" in it:
                    return [_file_entry(drive, path, it)]
            purl = body.get("@odata.nextLink")
        raise FileNotFoundError(f"file root not found: {folder}")
    url = f"{GRAPH}/drives/{drive}/root:/{path}:/children" if path else f"{GRAPH}/drives/{drive}/root/children"
    out = []
    while url:
        r = requests.get(url, headers=_headers(key), timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        for it in body.get("value", []):
            child_path = f"{path}/{it['name']}" if path else it["name"]
            if "folder" in it:
                if it["name"].strip().lower() in skip:
                    if skipped is not None:
                        skipped.append(f"{drive}:{child_path}")
                    continue
                if recurse:
                    out.extend(list_folder(f"{drive}:{child_path}", recurse=True, skip_folders=skip_folders,
                                           skipped=skipped, _top=False))
                continue
            out.append(_file_entry(drive, child_path, it))
        url = body.get("@odata.nextLink")
    return out


def _looks_like_file(path: str) -> bool:
    """A root names a file when its last segment ends in a type we know. "07.2 Plan" does not."""
    ext = os.path.splitext(path.rsplit("/", 1)[-1])[1].lower()
    return ext in READABLE_EXT or ext in IMAGE_EXT


def reader_for(ext: str) -> str | None:
    """"text" for a readable type, "image" for an image (arturo.py transcribes), else None."""
    if ext in READABLE_EXT:
        return "text"
    if ext in IMAGE_EXT:
        return "image"
    return None


def read_file_bytes(item: dict) -> bytes:
    if item.get("kind") == "list":
        return item["_text"].encode("utf-8")
    key = _provider_key(item["path"])
    url = f"{GRAPH}/drives/{item['drive_id']}/items/{item['item_id']}/content"
    r = requests.get(url, headers=_headers(key), timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r.content


def read_file_text(item: dict) -> str:
    """Plain text from every type in READABLE_EXT, or the rendered list. Raises on any type
    it cannot read and on any file it cannot open (password protected, corrupt, scanned PDF)."""
    if item.get("kind") == "list":
        return item["_text"]
    ext = item["ext"]
    if ext not in READABLE_EXT:
        raise NotImplementedError(f"no text extractor for {ext}: {item['name']}")
    data = read_file_bytes(item)
    if ext in TEXT_EXT:
        return data.decode("utf-8", errors="replace")
    if ext in DOCX_EXT:
        from docx import Document  # python-docx
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        text = "\n".join(parts)
        if not text.strip():
            text = _docx_shape_text(data)   # text boxes and flowchart shapes
        return text
    if ext in PPTX_EXT:
        return _pptx_text(data)
    if ext in XLSX_EXT:
        return _xlsx_text(data, item["name"])
    if ext in PDF_EXT:
        return _pdf_text(data, item["name"])
    if ext in ODT_EXT:
        return _odt_text(data)
    if ext in HTML_EXT:
        return _html_text(data)
    raise NotImplementedError(f"no text extractor for {ext}: {item['name']}")


def _docx_shape_text(data: bytes) -> str:
    """Every <w:t> run in the document body, including text boxes and drawn shapes, which
    python-docx paragraphs do not reach."""
    import re
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    runs = re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>|(\n)", xml)
    text = "".join(t or nl for t, nl in runs)
    lines = [" ".join(l.split()) for l in _unescape(text).splitlines()]
    return "\n".join(l for l in lines if l)


def embedded_images(item: dict, data: bytes | None = None) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Pictures inside a pdf/docx/odt/pptx, ready for the model: [(format, bytes)] with format
    png/jpeg/gif/webp, plus a list of picture formats that could not be converted. Small pictures
    (logos, icons) are left out. At most MAX_IMAGES_PER_FILE, in document order."""
    import zipfile
    from PIL import Image
    data = data if data is not None else read_file_bytes(item)
    ext = item["ext"]
    raw: list[tuple[str, bytes]] = []
    if ext == ".pdf":
        from pypdf import PdfReader
        for page in PdfReader(io.BytesIO(data)).pages:
            for im in page.images:
                raw.append((im.name, im.data))
    else:
        prefix = {".docx": "word/media/", ".odt": "Pictures/", ".pptx": "ppt/media/"}[ext]
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in sorted(z.namelist()):
                if n.startswith(prefix) and not n.endswith("/"):
                    raw.append((n, z.read(n)))
    out, unsupported = [], []
    for name, b in raw:
        if len(out) >= MAX_IMAGES_PER_FILE:
            break
        try:
            im = Image.open(io.BytesIO(b))
            im.load()
        except Exception:  # noqa: BLE001  emf/wmf and other formats Pillow cannot open
            unsupported.append(os.path.splitext(name)[1].lower() or "unknown")
            continue
        if min(im.size) < MIN_IMAGE_SIDE:
            continue
        fmt = (im.format or "").lower()
        fmt = "jpeg" if fmt == "jpg" else fmt
        if fmt in ("png", "jpeg", "gif", "webp") and len(b) <= MAX_IMAGE_BYTES and max(im.size) <= MAX_IMAGE_SIDE:
            out.append((fmt, b))
            continue
        im = im.convert("RGB")
        im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        for q in (85, 70, 55):
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q)
            if buf.tell() <= MAX_IMAGE_BYTES:
                break
        out.append(("jpeg", buf.getvalue()))
    return out, sorted(set(unsupported))


def _pptx_text(data: bytes) -> str:
    from pptx import Presentation  # python-pptx
    prs = Presentation(io.BytesIO(data))
    out = []

    def shape_text(sh, acc):
        if getattr(sh, "shape_type", None) == 6 and hasattr(sh, "shapes"):   # group
            for g in sh.shapes:
                shape_text(g, acc)
            return
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                acc.append(" | ".join(c.text.strip() for c in row.cells))
            return
        if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
            t = sh.text_frame.text.strip()
            if t:
                acc.append(t)

    for i, slide in enumerate(prs.slides, 1):
        acc = []
        for sh in slide.shapes:
            shape_text(sh, acc)
        if slide.has_notes_slide:
            n = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            if n:
                acc.append(f"Speaker notes: {n}")
        out.append(f"--- Slide {i} ---\n" + "\n".join(acc))
    return "\n\n".join(out)


def _xlsx_text(data: bytes, name: str) -> str:
    import zipfile
    from openpyxl import load_workbook
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except zipfile.BadZipFile:
        # An encrypted workbook is an OLE container, not a zip.
        raise ValueError(f"cannot open {name}: password protected or not a valid workbook")
    out = []
    for ws in wb.worksheets:
        rows, cut = [], False
        for n, row in enumerate(ws.iter_rows(values_only=True)):
            if n >= XLSX_MAX_ROWS:
                cut = True
                break
            cells = ["" if v is None else str(v).replace("\n", " ").strip() for v in row]
            if any(cells):
                while cells and not cells[-1]:
                    cells.pop()
                rows.append(" | ".join(cells))
        head = f"--- Sheet: {ws.title} ---"
        if cut:
            head += f" (first {XLSX_MAX_ROWS} rows only)"
        out.append(head + "\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(out)


def _pdf_text(data: bytes, name: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            raise ValueError(f"cannot open {name}: password protected PDF")
    pages = []
    for i, page in enumerate(reader.pages, 1):
        t = (page.extract_text() or "").strip()
        if t:
            pages.append(f"--- Page {i} ---\n{t}")
    if not pages:
        raise ValueError(f"no text layer in {name} (scanned PDF); not read")
    return "\n\n".join(pages)


def _odt_text(data: bytes) -> str:
    import re
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("content.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</text:(p|h)>", "\n", xml)
    xml = re.sub(r"<text:(tab|s)\b[^>]*/>", " ", xml)
    return _unescape(re.sub(r"<[^>]+>", "", xml)).strip()


def _html_text(data: bytes) -> str:
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.out, self.skip = [], 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip += 1
            elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "td", "th"):
                self.out.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style") and self.skip:
                self.skip -= 1

        def handle_data(self, d):
            if not self.skip:
                self.out.append(d)

    p = P()
    p.feed(data.decode("utf-8", errors="replace"))
    lines = [" ".join(l.split()) for l in "".join(p.out).splitlines()]
    return "\n".join(l for l in lines if l)


def _unescape(s: str) -> str:
    import html
    return html.unescape(s)


# ---- SharePoint list -> one Markdown document ----

def _get_all(url: str, key: str) -> list[dict]:
    out: list[dict] = []
    while url:
        r = requests.get(url, headers=_headers(key), timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        out.extend(body.get("value", []))
        url = body.get("@odata.nextLink")
    return out


def _cell(v) -> str:
    """Render one field value for a Markdown cell. Person/lookup values keep the display
    name only; email and ids are dropped. Pipes and newlines are flattened."""
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(_cell(x) for x in v if _cell(x))
    if isinstance(v, dict):
        return str(v.get("LookupValue") or v.get("Label") or v.get("DisplayName") or "")
    s = str(v)
    return s.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _list_item(root: str, drop_columns: list[str] | None = None) -> dict:
    site_id, list_name = _split_list(root)
    key = _provider_key(root)
    base = f"{GRAPH}/sites/{site_id}/lists/{quote(list_name)}"
    drop = {d.strip().lower() for d in (drop_columns or [])}

    columns = _get_all(f"{base}/columns", key)
    cols = [(c["name"], (c.get("displayName") or c["name"]).strip())
            for c in columns
            if not c.get("readOnly") and not c.get("hidden") and c["name"] not in _LIST_SKIP_COLUMNS]
    dropped = [h for internal, h in cols if h.lower() in drop or internal.lower() in drop]
    cols = [(internal, h) for internal, h in cols if h not in dropped]

    rows = _get_all(f"{base}/items?expand=fields", key)
    last_mod = max((r.get("lastModifiedDateTime", "") for r in rows), default="")
    read_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# {list_name} (SharePoint list)",
        "",
        f"Source: SharePoint list \"{list_name}\" on site {site_id}. {len(rows)} rows. "
        f"Read {read_at}. Newest row modified {last_mod or 'unknown'}. "
        "Columns are the list's own; person fields show display name only."
        + (f" Columns left out by the source map: {', '.join(dropped)}." if dropped else ""),
        "",
        "| " + " | ".join(h for _, h in cols) + " | Modified |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for r in rows:
        f = r.get("fields", {})
        cells = [_cell(f.get(internal)) for internal, _ in cols]
        cells.append(_cell(r.get("lastModifiedDateTime", "")))
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"

    return {
        "path": root,
        "name": f"{list_name} (SharePoint list).md",
        "item_id": "",
        "drive_id": "",
        "size": len(text.encode("utf-8")),
        "last_modified": last_mod,
        "ext": ".md",
        "kind": "list",
        "row_count": len(rows),
        "dropped_columns": dropped,
        "_text": text,
    }
