#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# hf-gated-downloader — resumable Hugging Face downloads under a folder-size cap.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License, distributed as LICENSE alongside this file, for more
# details, or <https://www.gnu.org/licenses/>.
"""Download a Hugging Face repository without exceeding a folder-size limit."""

from __future__ import annotations

import argparse
import curses
import hashlib
from http.client import IncompleteRead
import json
import os
import re
import signal
import ssl
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urljoin, urlparse
from urllib.request import Request, urlopen


HUB_URL = "https://huggingface.co"
# Blank on purpose: nothing should download until the user has named a repo.
# The panel shows the Repository field empty and refuses to start until it is set.
DEFAULT_REPOSITORY = ""
GIGABYTE = 1_000_000_000
CHUNK_SIZE = 1024 * 1024
SOCKET_TIMEOUT = 30
STALL_SECONDS = 90
USER_AGENT = "hf-gated-downloader/1.0"
DEFAULT_RETRIES = 8
RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}

TITLE = "HF GATED DOWNLOADER"
# Colour pairs. Only hues that survive both dark and light backgrounds are used,
# and every pair is optional: without colour support the attributes stay at zero.
PAIR_ACCENT = 1
PAIR_OK = 2
PAIR_WARN = 3
PAIR_NOTE = 4
PAIR_ALERT = 5
PALETTE = (
    curses.COLOR_CYAN,
    curses.COLOR_GREEN,
    curses.COLOR_YELLOW,
    curses.COLOR_MAGENTA,
    curses.COLOR_RED,
)
STATUS_STYLE: dict[str, tuple[str, int, int]] = {
    "PREPARING": ("○", PAIR_ACCENT, 0),
    "DOWNLOADING": ("●", PAIR_OK, 0),
    "WAITING": ("◐", PAIR_WARN, curses.A_REVERSE),
    "RETRYING": ("◑", PAIR_WARN, curses.A_REVERSE),
    "EXHAUSTED": ("▲", PAIR_ALERT, curses.A_REVERSE),
    "PAUSED": ("■", PAIR_NOTE, curses.A_REVERSE),
    "COMPLETE": ("✓", PAIR_OK, curses.A_REVERSE),
    "ERROR": ("✕", PAIR_ALERT, curses.A_REVERSE),
}
EIGHTHS = " ▏▎▍▌▋▊▉"
LEVELS = " ▁▂▃▄▅▆▇█"
RATE_WINDOW = 20.0
SAMPLE_SECONDS = 1.0
HISTORY_CELLS = 480


class DownloadError(RuntimeError):
    """A download cannot safely continue."""


class DownloadCancelled(RuntimeError):
    """The user requested a clean stop."""


class RetryableDownloadError(RuntimeError):
    """A temporary transport problem; retry from the retained partial file."""


@dataclass(frozen=True)
class Repository:
    repo_id: str
    repo_type: str
    revision: str
    subfolder: str = ""

    @property
    def api_kind(self) -> str:
        return {"model": "models", "dataset": "datasets", "space": "spaces"}[self.repo_type]

    @property
    def url_prefix(self) -> str:
        return "" if self.repo_type == "model" else f"{self.repo_type}s/"

    @property
    def name(self) -> str:
        """Folder-safe model name, e.g. unsloth/Kimi-K3 -> Kimi-K3."""

        candidate = self.repo_id.split("/")[-1]
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-.")
        return cleaned or re.sub(r"[^A-Za-z0-9._-]+", "-", self.repo_id).strip("-.") or "download"

    @property
    def label(self) -> str:
        prefix = "" if self.repo_type == "model" else f"{self.repo_type}s/"
        suffix = f"/{self.subfolder}" if self.subfolder else ""
        return f"{prefix}{self.repo_id}@{self.revision}{suffix}"


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    oid: str = ""      # git blob sha1 for ordinary files
    sha256: str = ""   # LFS digest, present on the large shards


# Declared after RemoteFile: these aliases are evaluated at import, not deferred.
BlockCallback = Callable[[int], None] | None
CheckCallback = Callable[[RemoteFile], None] | None


@dataclass
class State:
    status: str = "PREPARING"
    message: str = "Loading repository manifest…"
    repository: str = ""
    destination: str = ""
    total_files: int = 0
    complete_files: int = 0
    total_bytes: int = 0
    complete_bytes: int = 0
    current_path: str = "—"
    current_size: int = 0
    current_done: int = 0
    storage_used: int = 0
    storage_limit: int = 0
    resume_limit: int = 0
    waiting_reason: str = ""
    stalled_for: float = 0.0
    retry_limit: int = 0
    retry_attempt: int = 0
    reconciled: str = ""
    elapsed_start: float = field(default_factory=time.monotonic)


def parse_repository(value: str, revision_override: str | None, subfolder_override: str | None = None) -> Repository:
    """Accept org/repo or a Hugging Face model, dataset, or Space URL.

    A URL carrying ?show_file_info=<repo-relative file path> resolves to the
    folder holding that file, so a copied quant link downloads that quant only.
    An explicit --subfolder always wins over anything the URL implies.
    """

    raw = value.strip().rstrip("/")
    if not raw:
        raise argparse.ArgumentTypeError("repository cannot be empty")

    repo_type = "model"
    revision = "main"
    subfolder = ""
    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise argparse.ArgumentTypeError("repository URL must be hosted on huggingface.co")
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0] in {"datasets", "spaces"}:
            repo_type = "dataset" if parts.pop(0) == "datasets" else "space"
        if len(parts) < 2:
            raise argparse.ArgumentTypeError("repository URL must include an owner and repository name")
        repo_id = "/".join(parts[:2])
        if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"}:
            revision = parts[3]
            if len(parts) > 4:
                subfolder = "/".join(parts[4:])
        elif parsed.query and not subfolder:
            # A copied file link carries ?show_file_info=<repo-relative path>, e.g.
            # huggingface.co/unsloth/GLM-5.3-GGUF?show_file_info=UD-Q6_K_XL%2Fmodel.gguf.
            # The folder holding that file is the quant the user pointed at; treat
            # it as the subfolder so one quant downloads, not every quant (or the
            # repo-wide default) in the revision.
            pointers = parse_qs(parsed.query).get("show_file_info", [])
            if pointers:
                pointer = pointers[0].strip("/")
                if "/" in pointer:
                    subfolder = pointer.rsplit("/", 1)[0]
    else:
        parts = [part for part in raw.split("/") if part]
        if parts and parts[0] in {"datasets", "spaces"}:
            repo_type = "dataset" if parts.pop(0) == "datasets" else "space"
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "use owner/repository or a full Hugging Face repository URL"
            )
        repo_id = "/".join(parts)

    if revision_override:
        revision = revision_override.strip()
    if not revision:
        raise argparse.ArgumentTypeError("revision cannot be empty")
    if subfolder_override:
        subfolder = subfolder_override.strip().strip("/")
    return Repository(repo_id=repo_id, repo_type=repo_type, revision=revision, subfolder=subfolder)


def readable_size(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ("KB", "MB", "GB", "TB", "PB")
    amount = float(value)
    for unit in units:
        amount /= 1000
        if amount < 1000 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
    return f"{amount:.2f} PB"


def readable_rate(value: float) -> str:
    return f"{readable_size(int(value))}/s" if value >= 1 else "—"


def readable_span(seconds: float) -> str:
    """Coarse duration for an ETA; anything absurd degrades to an em dash."""

    if seconds <= 0 or seconds != seconds or seconds > 60 * 60 * 24 * 30:
        return "—"
    total = int(seconds)
    hours, minutes, rest = total // 3600, (total % 3600) // 60, total % 60
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {rest:02d}s"
    return f"{rest}s"


def clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def percent(value: float, maximum: float) -> str:
    return f"{min(100, int(value * 100 / maximum))}%" if maximum > 0 else "—"


def shorten_left(text: str, width: int) -> str:
    """Drop the head of a path so the file name stays visible."""

    if width <= 1:
        return text[: max(0, width)]
    return text if len(text) <= width else "…" + text[-(width - 1) :]


def span_between(left: str, right: str, width: int) -> str:
    """One line with `left` flush left and `right` flush right, `left` giving way."""

    room = max(0, width - len(right))
    if len(left) > room:
        left = f"{left[: room - 1]}…" if room else ""
    return left.ljust(room) + right


def folder_size(directory: Path) -> int:
    total = 0
    for root, _, files in os.walk(directory, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def safe_target(destination: Path, remote_path: str) -> Path:
    path = PurePosixPath(remote_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DownloadError(f"Hub returned an unsafe file path: {remote_path!r}")
    target = destination.joinpath(*path.parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as error:
        raise DownloadError(f"Refusing to write outside destination: {remote_path!r}") from error
    return target


def fetch_json(url: str, token: str | None) -> tuple[Any, str | None]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response), response.headers.get("Link")
    except HTTPError as error:
        if error.code in {401, 403}:
            raise DownloadError("Hub denied access. Export HF_TOKEN for this private or gated repository.") from error
        raise DownloadError(f"Hub API request failed ({error.code}): {url}") from error
    except URLError as error:
        raise DownloadError(f"Cannot reach Hugging Face: {error.reason}") from error


def next_link(link_header: str | None, current_url: str) -> str | None:
    if not link_header:
        return None
    match = re.search(r"<([^>]+)>;\s*rel=\"?next\"?", link_header)
    return urljoin(current_url, match.group(1)) if match else None


def list_files(repository: Repository, token: str | None, expand: bool = False) -> list[RemoteFile]:
    """List the revision. `expand` also fetches digests, which the Hub pages more slowly."""

    revision = quote(repository.revision, safe="")
    subfolder_path = f"/{quote(repository.subfolder, safe='/')}" if repository.subfolder else ""
    url = (
        f"{HUB_URL}/api/{repository.api_kind}/{repository.repo_id}/tree/{revision}{subfolder_path}"
        f"?recursive=true&expand={'true' if expand else 'false'}"
    )
    files: list[RemoteFile] = []
    while url:
        payload, links = fetch_json(url, token)
        if not isinstance(payload, list):
            raise DownloadError("Hub returned an unexpected repository manifest.")
        for item in payload:
            if item.get("type") != "file":
                continue
            path = item.get("path")
            size = item.get("size")
            if not isinstance(path, str) or not isinstance(size, int) or size < 0:
                raise DownloadError("Hub manifest includes a file without a usable size.")
            lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
            sha256 = str(lfs.get("oid") or "").removeprefix("sha256:")
            files.append(
                RemoteFile(path=path, size=size, oid=str(item.get("oid") or ""), sha256=sha256)
            )
        url = next_link(links, url)
    if not files:
        raise DownloadError("No files found in this revision.")
    return files


# Files a runner needs that sit outside a quant folder. A GGUF normally carries
# its own tokenizer and chat template, but multi-modal builds keep the vision
# projector (mmproj) — and some repos keep tokenizer-side config — at the root,
# next to the quant folders rather than inside one.
RUNTIME_EXACT_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "preprocessor_config.json",
    "image_processor_config.json",
    "video_processor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "template",
    "params",
}


def is_runtime_extra(path: str) -> bool:
    """True for a file the model needs at run time but that lives outside the
    quant folder. Quant shards from sibling folders never match."""

    name = path.rsplit("/", 1)[-1].lower()
    if name in RUNTIME_EXACT_NAMES:
        return True
    return name.startswith("mmproj") or "chat_template" in name


@dataclass
class Audit:
    """What the destination folder holds, measured against the repository manifest."""

    adopted: list[RemoteFile] = field(default_factory=list)     # present and trusted
    recorded: list[RemoteFile] = field(default_factory=list)    # finished earlier, since moved away
    mismatched: list[RemoteFile] = field(default_factory=list)  # present at the wrong size
    corrupt: list[RemoteFile] = field(default_factory=list)     # right size, wrong digest
    pending: list[RemoteFile] = field(default_factory=list)     # never fetched
    partials: list[str] = field(default_factory=list)           # interrupted .part files
    stale: list[str] = field(default_factory=list)              # .part files from another revision
    stale_bytes: int = 0
    unknown: list[str] = field(default_factory=list)            # folder contents outside the manifest

    @property
    def settled(self) -> list[RemoteFile]:
        return self.adopted + self.recorded

    @property
    def outstanding(self) -> list[RemoteFile]:
        return self.mismatched + self.corrupt + self.pending

    def summary(self) -> str:
        parts = []
        if self.adopted:
            # Rendered under an "Already had" label, so no second "already" here.
            parts.append(f"{len(self.adopted):,} in the folder")
        if self.recorded:
            parts.append(f"{len(self.recorded):,} completed earlier and moved away")
        if self.mismatched:
            parts.append(f"{len(self.mismatched):,} wrong size")
        if self.corrupt:
            parts.append(f"{len(self.corrupt):,} failed the digest check")
        if self.partials:
            parts.append(f"{len(self.partials):,} partly downloaded")
        if self.stale:
            parts.append(f"{readable_size(self.stale_bytes)} of partials from another revision")
        if self.unknown:
            parts.append(f"{len(self.unknown):,} not part of this repository")
        return "  ·  ".join(parts)


def hash_file(path: Path, digest: Any, on_block: BlockCallback = None) -> str:
    """Digest a file a block at a time so the caller stays responsive on huge shards."""

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
            if on_block:
                on_block(len(block))
    return digest.hexdigest()


def digest_matches(path: Path, remote: RemoteFile, on_block: BlockCallback = None) -> bool:
    """Compare against whichever digest the Hub published for this file."""

    expected = remote.sha256.lower()
    if len(expected) == 64:
        return hash_file(path, hashlib.sha256(), on_block) == expected
    expected = remote.oid.lower()
    if len(expected) == 40:
        # Git hashes the header plus the contents, which is what the Hub stores.
        return hash_file(path, hashlib.sha1(b"blob %d\0" % remote.size), on_block) == expected
    return True  # nothing published to compare against; the size already agreed


def audit_destination(
    files: list[RemoteFile],
    destination: Path,
    completed: dict[str, int],
    verify: bool,
    partial_suffix: str,
    on_check: CheckCallback = None,
    on_block: BlockCallback = None,
) -> Audit:
    """Sort every manifest entry by what the folder and the session store already hold."""

    result = Audit()
    manifest = {item.path for item in files}
    for remote in files:
        target = safe_target(destination, remote.path)
        if target.with_name(target.name + partial_suffix).exists():
            result.partials.append(remote.path)
        try:
            actual = target.stat().st_size if target.is_file() else -1
        except OSError:
            actual = -1
        if actual == remote.size:
            if on_check:
                on_check(remote)
            if verify and not digest_matches(target, remote, on_block):
                result.corrupt.append(remote)
            else:
                result.adopted.append(remote)
        elif actual >= 0:
            result.mismatched.append(remote)
        elif completed.get(remote.path) == remote.size:
            result.recorded.append(remote)
        else:
            result.pending.append(remote)
    for root, _, names in os.walk(destination, followlinks=False):
        for name in names:
            full = Path(root) / name
            relative = full.relative_to(destination).as_posix()
            if relative.endswith(".part"):
                # A partial keyed to a different revision is dead weight that still
                # counts against the folder limit, so surface it rather than hide it.
                if not relative.endswith(partial_suffix):
                    result.stale.append(relative)
                    try:
                        result.stale_bytes += full.stat().st_size
                    except OSError:
                        pass
            elif relative not in manifest:
                result.unknown.append(relative)
    return result


class SessionStore:
    """Persist completed paths outside the destination so moved files stay complete."""

    def __init__(self, destination: Path, repository: Repository):
        identity = f"{destination.resolve()}\0{repository.label}".encode()
        digest = hashlib.sha256(identity).hexdigest()[:16]
        state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        self.path = state_home / "hf-gated-downloader" / f"{digest}.json"
        self.identity = repository.label
        self.completed: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text())
            if payload.get("repository") == self.identity:
                self.completed = {
                    path: int(size)
                    for path, size in payload.get("completed", {}).items()
                    if isinstance(path, str) and isinstance(size, int)
                }
        except (OSError, ValueError, TypeError):
            return

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"repository": self.identity, "completed": self.completed}, indent=2) + "\n"
        )
        os.replace(temporary, self.path)

    def mark_complete(self, remote_file: RemoteFile) -> None:
        self.completed[remote_file.path] = remote_file.size
        self.save()

    def mark_many(self, remote_files: list[RemoteFile]) -> None:
        """Adopt a batch in one write rather than one write per file."""

        if not remote_files:
            return
        for remote_file in remote_files:
            self.completed[remote_file.path] = remote_file.size
        self.save()


class TokenStore:
    """Remember the access token between runs so it is typed once, not every launch."""

    def __init__(self) -> None:
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        self.path = config_home / "hf-gated-downloader" / "token"

    def load(self) -> str:
        """Our own saved token, falling back to one an `hf auth login` already wrote."""

        for candidate in (self.path, huggingface_cli_token_path()):
            try:
                token = candidate.read_text().strip()
            except OSError:
                continue
            if token:
                return token
        return ""

    def remember(self, token: str) -> None:
        """Store a token the user typed, or forget the stored one when blanked."""

        token = token.strip()
        if not token:
            self.forget()
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            # Create at 0600 rather than chmod after: the secret is never
            # briefly readable by anyone else on the machine.
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                handle.write(token + "\n")
            os.replace(temporary, self.path)
        except OSError:
            # A token that cannot be saved still works for this run.
            return

    def forget(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            return


def huggingface_cli_token_path() -> Path:
    """Where `hf auth login` keeps its token, so an existing login is picked up."""

    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "huggingface"
    return base / "token"


@dataclass
class Settings:
    """Every knob the downloader has. Edited in the TUI, never required as a flag."""

    repository: str = DEFAULT_REPOSITORY
    revision: str = ""
    subfolder: str = ""
    output: str = "."
    use_subdir: bool = True
    dir_name: str = ""
    max_gb: float = 100.0
    resume_gb: float = 60.0
    poll_seconds: int = 60
    retries: int = DEFAULT_RETRIES
    token: str = ""
    verify: bool = False
    retry_missing: bool = False

    @property
    def max_bytes(self) -> int:
        return int(self.max_gb * GIGABYTE)

    @property
    def resume_bytes(self) -> int:
        return int(self.resume_gb * GIGABYTE)

    def parse(self) -> Repository:
        return parse_repository(self.repository, self.revision or None, self.subfolder or None)

    def destination(self) -> Path:
        base = Path(self.output).expanduser()
        if not self.use_subdir:
            return base
        return base / (self.dir_name.strip() or self.parse().name)

    def normalize(self) -> None:
        """Keep the resume gate meaningfully below the ceiling as the ceiling moves."""

        if self.max_gb > 0 and self.resume_gb >= self.max_gb:
            self.resume_gb = round(self.max_gb * 0.6, 2)
        self.poll_seconds = max(1, min(60, self.poll_seconds))

    def validate(self) -> str:
        try:
            self.parse()
        except argparse.ArgumentTypeError as error:
            return str(error)
        if self.max_gb <= 0:
            return "Max folder size must be greater than zero."
        if self.resume_gb <= 0:
            return "Resume threshold must be greater than zero."
        if self.resume_gb >= self.max_gb:
            return "Resume threshold must be below the max folder size."
        if self.retries < 0:
            return "Retries cannot be negative."
        name = self.dir_name.strip()
        if name and (name in {".", ".."} or "/" in name or os.sep in name):
            return "Folder name must be a single folder, not a path."
        if not str(self.output).strip():
            return "Parent folder cannot be empty."
        return ""


@dataclass(frozen=True)
class Field:
    label: str
    attr: str
    kind: str  # text | number | int | bool
    hint: str
    step: float = 1.0
    live: bool = False
    secret: bool = False


FIELDS: tuple[Field, ...] = (
    Field("Repository", "repository", "text", "owner/repo or any huggingface.co URL"),
    Field("Revision", "revision", "text", "branch, tag, or commit — blank uses the URL or main"),
    Field("Subfolder", "subfolder", "text", "download only this path — blank takes everything"),
    Field("Parent folder", "output", "text", "where the model folder is created"),
    Field("Model subfolder", "use_subdir", "bool", "put files in a folder named after the model"),
    Field("Folder name", "dir_name", "text", "blank uses the model name from the URL"),
    Field("Max folder size", "max_gb", "number", "GB this folder may never exceed", step=5, live=True),
    Field("Resume below", "resume_gb", "number", "once full, wait until the folder drops under this", step=5, live=True),
    Field("Poll interval", "poll_seconds", "int", "seconds between storage checks (1-60)", step=5, live=True),
    Field("Retries", "retries", "int", "attempts after a dropped connection", step=1, live=True),
    Field("Access token", "token", "text", "needed for gated repos; saved for next launch", secret=True),
    Field("Verify existing", "verify", "bool", "checksum files already in the folder instead of trusting their size"),
    Field("Re-download moved", "retry_missing", "bool", "fetch files completed earlier but since moved away"),
)

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SOURCE", ("repository", "revision", "subfolder")),
    ("DESTINATION", ("output", "use_subdir", "dir_name")),
    ("LIMITS", ("max_gb", "resume_gb", "poll_seconds", "retries")),
    ("ACCESS", ("token",)),
    ("INTEGRITY", ("verify", "retry_missing")),
)
# Validation strings that do not open with a field label.
ERROR_HINTS: tuple[tuple[str, str], ...] = (
    ("resume threshold", "resume_gb"),
    ("revision", "revision"),
    ("subfolder", "subfolder"),
)


def group_of(attr: str) -> str:
    for name, attrs in GROUPS:
        if attr in attrs:
            return name
    return "OTHER"


@dataclass(frozen=True)
class Line:
    """One rendered row. Higher priority is shed first when the screen is short."""

    text: str
    attr: int = 0
    priority: int = 0


class Terminal:
    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdout.isatty() and sys.stdin.isatty()
        self.screen: curses.window | None = None
        self.paused = False
        self.stop_requested = False
        self.retry_adjustment = 0
        self.reset_requested = False
        self.configure_requested = False
        self.last_render = 0.0
        self.colors = False
        self.samples: list[tuple[float, int]] = []
        self.history: list[float] = []
        self.bucket: tuple[float, int] | None = None

    def open(self) -> None:
        if not self.enabled or self.screen:
            return
        try:
            self.screen = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.screen.keypad(True)
            self.screen.nodelay(True)
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            self._start_colors()
        except curses.error:
            self.close()
            self.enabled = False

    def _start_colors(self) -> None:
        """Colour is a bonus: a monochrome terminal keeps every attribute at zero."""

        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
            for index, colour in enumerate(PALETTE, start=1):
                curses.init_pair(index, colour, -1)
        except curses.error:
            return
        self.colors = True

    def _pair(self, index: int) -> int:
        if not self.colors:
            return 0
        try:
            return curses.color_pair(index)
        except curses.error:
            return 0

    def close(self) -> None:
        if not self.screen:
            return
        try:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        except curses.error:
            pass
        self.screen = None

    def _add(self, row: int, text: str, attribute: int = 0) -> None:
        if not self.screen:
            return
        height, width = self.screen.getmaxyx()
        if row < 0 or row >= height or width < 2:
            return
        try:
            self.screen.addnstr(row, 0, text, max(0, width - 1), attribute)
        except curses.error:
            pass

    @staticmethod
    def _bar(value: int, maximum: int, width: int = 38) -> str:
        """Filled bar with an eighth-of-a-cell leading edge for slow-moving totals."""

        width = max(1, width)
        ratio = min(1.0, max(0.0, value / maximum)) if maximum > 0 else 0.0
        cells = ratio * width
        filled = min(width, int(cells))
        bar = "█" * filled
        if filled < width:
            edge = int((cells - filled) * 8)
            bar += EIGHTHS[edge] if edge else "░"
            bar += "░" * (width - filled - 1)
        return bar

    def _rule(self, title: str, width: int) -> Line:
        head = f"  {title} " if title else "  "
        return Line(
            head + "─" * max(0, width - 1 - len(head)),
            self._pair(PAIR_ACCENT) | curses.A_DIM,
            priority=4,
        )

    @staticmethod
    def _window(lines: list[Line], focus: int, capacity: int) -> list[Line]:
        """Scroll a too-tall list so the focused row stays on screen, with edge markers."""

        if capacity <= 0 or len(lines) <= capacity:
            return lines
        start = min(max(0, focus - capacity // 2), len(lines) - capacity)
        visible = list(lines[start : start + capacity])
        if start > 0:
            visible[0] = Line(f"  ↑ {start} more above", curses.A_DIM)
        if start + capacity < len(lines):
            visible[-1] = Line(f"  ↓ {len(lines) - start - capacity} more below", curses.A_DIM)
        return visible

    @staticmethod
    def _flow(lines: list[Line], capacity: int) -> list[Line]:
        """Shed the least important rows, from the bottom up, until the body fits."""

        kept = list(lines)
        while len(kept) > capacity:
            weakest = max(line.priority for line in kept)
            if weakest == 0:
                return kept[: max(0, capacity)]
            for position in range(len(kept) - 1, -1, -1):
                if kept[position].priority == weakest:
                    del kept[position]
                    break
        return kept

    @staticmethod
    def _controls(hints: list[str], width: int) -> str:
        """Fit as many whole hints as the width allows; never cut one in half."""

        kept: list[str] = []
        for hint in hints:
            candidate = "  " + "   ".join(kept + [hint])
            if len(candidate) > width - 1 and kept:
                break
            kept.append(hint)
        return "  " + "   ".join(kept)

    def _paint(self, body: list[Line], footer: str) -> None:
        """Draw a flowing body plus a footer pinned to the last row. One write per row."""

        if not self.screen:
            return
        try:
            height, width = self.screen.getmaxyx()
            self.screen.erase()
        except curses.error:
            return
        if height >= 5 and width >= 20:
            capacity = height - 2
        else:
            capacity, footer = max(1, height), ""
        for row, line in enumerate(self._flow(body, capacity)):
            self._add(row, line.text, line.attr)
        if footer:
            self._add(height - 2, "  " + "─" * max(0, width - 3), self._pair(PAIR_ACCENT) | curses.A_DIM)
            self._add(height - 1, footer, curses.A_BOLD)
        try:
            self.screen.refresh()
        except curses.error:
            pass

    def _masthead(self, width: int, right: str, subtitle: str) -> list[Line]:
        return [
            Line(span_between(f"  {TITLE}", f"{right}  ", width - 1), self._pair(PAIR_ACCENT) | curses.A_BOLD),
            Line("  " + "─" * max(0, width - 3), self._pair(PAIR_ACCENT) | curses.A_DIM, priority=4),
            Line(subtitle, self._pair(PAIR_ACCENT) | curses.A_DIM, priority=3),
        ]

    def _observe(self, state: State, now: float) -> float:
        """Transfer rate over a short trailing window, so it tracks the live link."""

        moved = state.complete_bytes + state.current_done
        self.samples.append((now, moved))
        while len(self.samples) > 2 and self.samples[0][0] < now - RATE_WINDOW:
            self.samples.pop(0)
        # One graph column per second, independent of how often the screen repaints.
        if self.bucket is None:
            self.bucket = (now, moved)
        elif now - self.bucket[0] >= SAMPLE_SECONDS:
            self.history.append(max(0.0, (moved - self.bucket[1]) / (now - self.bucket[0])))
            del self.history[:-HISTORY_CELLS]
            self.bucket = (now, moved)
        first_time, first_bytes = self.samples[0]
        span = now - first_time
        return max(0.0, (moved - first_bytes) / span) if span >= 1.0 else 0.0

    @staticmethod
    def _chart(values: list[float], width: int, height: int, peak: float) -> list[str]:
        """btop-style column graph: newest sample on the right, one row per band."""

        window = values[-width:]
        pad = " " * (width - len(window))
        rows: list[str] = []
        for row in range(height):
            ceiling = peak * (height - row) / height
            floor = peak * (height - row - 1) / height
            cells = []
            for value in window:
                if value >= ceiling:
                    cells.append("█")
                elif value <= floor:
                    cells.append(" ")
                else:
                    cells.append(LEVELS[max(1, min(8, int((value - floor) / (ceiling - floor) * 8)))])
            rows.append(pad + "".join(cells))
        return rows

    def pump(self, state: State, force: bool = False) -> None:
        if not self.enabled or not self.screen:
            return
        key = self.screen.getch()
        if key in (ord("q"), ord("Q")):
            self.stop_requested = True
        elif key in (ord("p"), ord("P"), ord(" ")):
            self.paused = not self.paused
        elif key in (ord("+"), ord("="), curses.KEY_RIGHT):
            self.retry_adjustment += 1
        elif key in (ord("-"), ord("_"), curses.KEY_LEFT):
            self.retry_adjustment -= 1
        elif key in (ord("r"), ord("R")):
            self.reset_requested = True
        elif key in (ord("c"), ord("C")):
            self.configure_requested = True
        elif key == curses.KEY_RESIZE:
            force = True
        now = time.monotonic()
        if not force and now - self.last_render < 0.1:
            return
        self.last_render = now
        height, width = self.screen.getmaxyx()
        rate = self._observe(state, now)
        footer = self._controls(
            [
                f"[p] {'resume' if self.paused else 'pause'}",
                "[q] stop",
                "[c] settings",
                "[+/-] retries",
                "[r] reset",
            ],
            width,
        )
        self._paint(self._transfer_body(state, height, width, now, rate), footer)

    def _transfer_body(self, state: State, height: int, width: int, now: float, rate: float) -> list[Line]:
        usable = max(16, width - 3)
        bar_width = max(8, min(96, usable - 20))
        label = 12 if usable >= 40 else 9
        ok, warn, alert = self._pair(PAIR_OK), self._pair(PAIR_WARN), self._pair(PAIR_ALERT)
        glyph, pair, extra = STATUS_STYLE.get(state.status, ("·", PAIR_ACCENT, 0))
        status = f"  {glyph} {state.status:<11}  {state.message}"
        if extra & curses.A_REVERSE:
            status = status[: width - 1].ljust(width - 1)
        body = self._masthead(width, f"elapsed {clock(now - state.elapsed_start)}", "")[:2]
        body.append(Line(status, self._pair(pair) | curses.A_BOLD | extra))

        moved = state.complete_bytes + state.current_done
        if state.status == "COMPLETE":
            body.append(
                Line(
                    f"  {state.complete_files:,} files  ·  {readable_size(state.complete_bytes)}  ·  "
                    f"finished in {readable_span(now - state.elapsed_start)}",
                    ok | curses.A_BOLD,
                    priority=1,
                )
            )
        if state.status == "EXHAUSTED":
            body.append(
                Line(
                    f"  [r] grant {state.retry_limit} more retries   "
                    f"[+/-] change the limit   [c] settings   [q] stop",
                    alert | curses.A_BOLD,
                )
            )
        if state.waiting_reason:
            body.append(Line(f"  ! {shorten_left(state.waiting_reason, usable - 2)}", warn, priority=1))
        if state.stalled_for > 3:
            body.append(
                Line(
                    f"  ! link idle for {state.stalled_for:.0f}s — dropped and resumed at {STALL_SECONDS}s",
                    warn,
                    priority=1,
                )
            )

        body.append(Line("", priority=5))
        body.append(self._rule("REPOSITORY", width))
        body.append(Line(f"  {'Source':<{label}}{shorten_left(state.repository, usable - label)}", priority=2))
        body.append(Line(f"  {'Folder':<{label}}{shorten_left(state.destination, usable - label)}", priority=2))
        if state.reconciled:
            body.append(
                Line(f"  {'Already had':<{label}}{shorten_left(state.reconciled, usable - label)}", ok, priority=3)
            )

        body.append(Line("", priority=5))
        body.append(self._rule("OVERALL", width))
        body.append(
            Line(
                f"  {'Files':<{label}}{state.complete_files:,} of {state.total_files:,}"
                f"   {percent(state.complete_files, state.total_files)}",
                priority=2,
            )
        )
        body.append(
            Line(
                span_between(
                    f"  {'Bytes':<{label}}{readable_size(moved)} of {readable_size(state.total_bytes)}",
                    percent(moved, state.total_bytes),
                    usable,
                ),
                priority=1,
            )
        )
        body.append(Line(f"  {self._bar(moved, state.total_bytes, bar_width)}", ok, priority=1))

        body.append(Line("", priority=5))
        body.append(self._rule("SPEED", width))
        remaining = max(0, state.total_bytes - moved)
        eta = readable_span(remaining / rate) if rate >= 1 else "—"
        peak = max(self.history) if self.history else 0.0
        elapsed = max(1.0, now - state.elapsed_start)
        head = f"  {'Rate':<{label}}{readable_rate(rate)}"
        if usable >= 60:
            head += f"   peak {readable_rate(peak)}"
        if usable >= 76:
            head += f"   avg {readable_rate(moved / elapsed)}"
        body.append(Line(span_between(head, f"ETA {eta}", usable), curses.A_BOLD, priority=1))
        rows = max(0, min(5, (height - 18) // 3)) if usable >= 40 else 0
        if rows and peak > 0:
            for graph in self._chart(self.history, usable, rows, peak):
                body.append(Line(f"  {graph}", self._pair(pair), priority=2))
            body.append(
                Line(
                    span_between(
                        f"  {'':<{label}}last {min(len(self.history), usable)}s",
                        f"full scale {readable_rate(peak)}",
                        usable,
                    ),
                    curses.A_DIM,
                    priority=5,
                )
            )

        if state.current_size or state.current_path not in ("", "—"):
            body.append(Line("", priority=5))
            body.append(self._rule("CURRENT FILE", width))
            body.append(Line(f"  {shorten_left(state.current_path, usable)}", curses.A_BOLD, priority=1))
            if state.current_size:
                detail = f"{readable_size(state.current_done)} of {readable_size(state.current_size)}"
                inner = max(8, min(bar_width, usable - len(detail) - 10))
                body.append(
                    Line(
                        span_between(
                            f"  {self._bar(state.current_done, state.current_size, inner)}"
                            f"  {percent(state.current_done, state.current_size):>4}",
                            detail,
                            usable,
                        ),
                        ok,
                        priority=1,
                    )
                )
            attempt = (
                f"attempt {state.retry_attempt} of {state.retry_limit + 1}"
                if state.retry_attempt > 1
                else "no retries in flight"
            )
            body.append(Line(f"  {'Retries':<{label}}limit {state.retry_limit}  ·  {attempt}", priority=2))

        body.append(Line("", priority=5))
        body.append(self._rule("STORAGE GATE", width))
        headroom = state.storage_limit - state.storage_used
        tail = f"headroom {readable_size(headroom)}" if headroom >= 0 else f"over by {readable_size(-headroom)}"
        fill = state.storage_used / state.storage_limit if state.storage_limit else 0.0
        gate = ok if fill < 0.75 else warn if fill < 0.95 else alert
        body.append(
            Line(
                span_between(
                    f"  {'Folder':<{label}}{readable_size(state.storage_used)} of "
                    f"{readable_size(state.storage_limit)}   {percent(state.storage_used, state.storage_limit)}",
                    tail if usable >= 60 else "",
                    usable,
                ),
                priority=1,
            )
        )
        body.append(Line(f"  {self._bar(state.storage_used, state.storage_limit, bar_width)}", gate, priority=1))
        body.append(
            Line(
                f"  {'Resume at':<{label}}below {readable_size(state.resume_limit)} once the ceiling is hit",
                curses.A_DIM,
                priority=3,
            )
        )
        body.append(
            Line(
                "  Completed files stay recorded even after you move them away.",
                curses.A_DIM,
                priority=4,
            )
        )
        return body


    @staticmethod
    def _display(settings: Settings, field: Field) -> str:
        value = getattr(settings, field.attr)
        if field.kind == "bool":
            return "yes" if value else "no"
        if field.secret:
            return f"set ({len(str(value))} characters)" if value else "not set"
        if field.kind == "number":
            return f"{float(value):g} GB"
        if field.kind == "int":
            return f"{int(value)} seconds" if field.attr == "poll_seconds" else str(int(value))
        return str(value) if str(value) else "—"

    @staticmethod
    def _nudge(settings: Settings, field: Field, direction: int) -> str:
        value = getattr(settings, field.attr)
        if field.kind == "bool":
            setattr(settings, field.attr, not value)
        elif field.kind == "number":
            setattr(settings, field.attr, max(0.5, round(float(value) + direction * field.step, 2)))
        elif field.kind == "int":
            setattr(settings, field.attr, max(0, int(value) + int(direction * field.step)))
        else:
            return ""
        settings.normalize()
        return ""

    @staticmethod
    def _commit(settings: Settings, field: Field, text: str) -> str:
        raw = text.strip()
        if field.kind == "number":
            try:
                setattr(settings, field.attr, max(0.01, float(raw)))
            except ValueError:
                return f"{field.label} needs a number."
        elif field.kind == "int":
            try:
                setattr(settings, field.attr, max(0, int(float(raw))))
            except ValueError:
                return f"{field.label} needs a whole number."
        else:
            setattr(settings, field.attr, raw)
        settings.normalize()
        return ""

    def _render_settings(
        self,
        settings: Settings,
        fields: tuple[Field, ...],
        index: int,
        editing: bool,
        pristine: bool,
        buffer: str,
        error: str,
        running: bool,
    ) -> None:
        if not self.screen:
            return
        width = self.screen.getmaxyx()[1]
        usable = max(16, width - 3)
        column = 19 if usable >= 44 else 13
        note = (
            "Held mid-transfer — these are the only settings that can move now"
            if running
            else "Everything is configured here — no command-line flags needed"
        )
        body = self._masthead(width, "LIVE SETTINGS" if running else "SETTINGS", f"  {note}")

        culprit = self._error_attr(error) if error else ""
        placed = False
        last_destination = ""
        for field in fields:
            if group_of(field.attr) == "DESTINATION":
                last_destination = field.attr
        group = ""
        focus = 0
        for position, field in enumerate(fields):
            name = group_of(field.attr)
            if name != group:
                group = name
                body.append(Line("", priority=5))
                rule = self._rule(name, width)
                body.append(Line(rule.text, rule.attr))
            selected = position == index
            marker = "▸" if selected else " "
            if selected and editing:
                shown = f"{buffer}▌"
                attribute = curses.A_BOLD | (curses.A_UNDERLINE if pristine else 0)
            else:
                shown = self._display(settings, field)
                attribute = curses.A_REVERSE if selected else 0
            if selected:
                focus = len(body)
            text = f"  {marker} {field.label:<{column}}{shorten_left(shown, usable - column - 4)}"
            # Setup mode flags which knobs stay reachable once the transfer starts.
            body.append(Line(span_between(text, "live" if field.live and not running else "", usable), attribute))
            if error and field.attr == culprit:
                body.append(Line(f"    ▲ {error}", self._pair(PAIR_ALERT) | curses.A_BOLD))
                placed = True
            if field.attr == last_destination:
                try:
                    destination = settings.destination()
                except argparse.ArgumentTypeError:
                    destination = Path(settings.output)
                if not destination.is_absolute():
                    destination = Path.cwd() / destination
                body.append(
                    Line(
                        f"    ↳ files land in {shorten_left(str(destination), usable - 20)}",
                        self._pair(PAIR_OK),
                        priority=2,
                    )
                )
        body.append(Line("", priority=5))
        body.append(Line(f"  {shorten_left(fields[index].hint, usable)}", self._pair(PAIR_ACCENT), priority=1))
        if error and not placed:
            body.append(Line(f"  ▲ {shorten_left(error, usable - 2)}", self._pair(PAIR_ALERT) | curses.A_BOLD))
        if editing:
            hints = ["[enter] confirm", "[esc] cancel"]
            hints += ["[type] replace", "[backspace] edit"] if pristine else ["[type] edit"]
        elif running:
            hints = ["[esc] resume", "[q] stop", "[↑/↓] move", "[enter] edit", "[+/-] adjust"]
        else:
            hints = ["[s] start", "[q] quit", "[↑/↓] move", "[enter] edit", "[+/-] adjust"]
        controls = self._controls(hints, width)
        try:
            height = self.screen.getmaxyx()[0]
        except curses.error:
            height = len(body) + 2
        # Windowed rather than shed: dropping a group rule would silently reparent
        # its fields under the group above, and the focused row must stay visible.
        capacity = height - 2 if height >= 5 and width >= 20 else max(1, height)
        self._paint(self._window(body, focus, capacity), controls)

    @staticmethod
    def _error_attr(error: str) -> str:
        """Point a validation message at the field it came from, for inline feedback."""

        for field in FIELDS:
            if error.startswith(field.label):
                return field.attr
        lowered = error.lower()
        for needle, attr in ERROR_HINTS:
            if needle in lowered:
                return attr
        return "repository"

    def configure(self, settings: Settings, running: bool = False) -> bool:
        """Edit settings in place. Returns False when the user backs out entirely."""

        if not self.enabled or not self.screen:
            return True
        fields = tuple(item for item in FIELDS if item.live) if running else FIELDS
        index = 0
        editing = False
        pristine = True
        buffer = ""
        error = ""
        while True:
            self._render_settings(settings, fields, index, editing, pristine, buffer, error, running)
            key = self.screen.getch()
            if key == -1:
                time.sleep(0.02)
                continue
            field = fields[index]
            if editing:
                if key in (10, 13, curses.KEY_ENTER):
                    error = self._commit(settings, field, buffer)
                    editing = bool(error)
                elif key == 27:
                    editing, buffer, error = False, "", ""
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    buffer, pristine = buffer[:-1], False
                elif 32 <= key < 127:
                    # The old value arrives pre-selected: the first keystroke
                    # replaces it, backspace keeps it and edits from the end.
                    buffer = chr(key) if pristine else buffer + chr(key)
                    pristine = False
                continue
            if key in (curses.KEY_UP, ord("k")):
                index, error = (index - 1) % len(fields), ""
            elif key in (curses.KEY_DOWN, ord("j"), 9):
                index, error = (index + 1) % len(fields), ""
            elif key in (curses.KEY_RIGHT, ord("+"), ord("=")):
                error = self._nudge(settings, field, 1)
            elif key in (curses.KEY_LEFT, ord("-"), ord("_")):
                error = self._nudge(settings, field, -1)
            elif key in (10, 13, curses.KEY_ENTER):
                if field.kind == "bool":
                    self._nudge(settings, field, 1)
                else:
                    value = getattr(settings, field.attr)
                    editing, pristine, error = True, True, ""
                    buffer = "" if field.secret else (f"{value:g}" if field.kind == "number" else str(value))
            elif key == 27 and running:
                error = settings.validate()
                if not error:
                    return True
            elif key in (ord("s"), ord("S")) and not running:
                error = settings.validate()
                if not error:
                    return True
            elif key in (ord("q"), ord("Q")):
                if running:
                    self.stop_requested = True
                return False


class Downloader:
    def __init__(self, settings: Settings, terminal: Terminal):
        self.settings = settings
        self.repository = settings.parse()
        self.destination = settings.destination()
        self.token = settings.token or None
        self.terminal = terminal
        self.stop_requested = False
        self.limit_was_reached = False
        # Support files pulled from outside the subfolder so the download is
        # runnable; reported at the end so their arrival is never a surprise.
        self.supporting: list[str] = []
        # Keying partials to the revision stops a stale .part from a different
        # revision being resumed into a file it does not belong to.
        stamp = hashlib.sha1(self.repository.label.encode()).hexdigest()[:8]
        self.partial_suffix = f".{stamp}.part"

    # Limits are read live so the settings panel can move them mid-transfer.
    @property
    def max_bytes(self) -> int:
        return self.settings.max_bytes

    @property
    def resume_bytes(self) -> int:
        return self.settings.resume_bytes

    @property
    def poll_seconds(self) -> int:
        return self.settings.poll_seconds

    @property
    def retry_missing(self) -> bool:
        return self.settings.retry_missing

    @property
    def retries(self) -> int:
        return self.settings.retries

    @retries.setter
    def retries(self, value: int) -> None:
        self.settings.retries = max(0, value)

    def _sync_limits(self, state: State) -> None:
        state.storage_limit = self.max_bytes
        state.resume_limit = self.resume_bytes
        state.retry_limit = self.retries

    def _apply_requests(self, state: State) -> None:
        """Act on keys the screen collected. Also runs while paused, so [c] works there."""

        if self.terminal.retry_adjustment:
            self.retries = self.retries + self.terminal.retry_adjustment
            self.terminal.retry_adjustment = 0
            self._sync_limits(state)
        if self.terminal.configure_requested:
            self.terminal.configure_requested = False
            self.terminal.configure(self.settings, running=True)
            self._sync_limits(state)
            self.terminal.pump(state, force=True)

    def _check_controls(self, state: State) -> None:
        self.terminal.pump(state)
        self._apply_requests(state)
        if self.stop_requested or self.terminal.stop_requested:
            raise DownloadCancelled("Stopped by user; partial files are retained for resuming.")
        while self.terminal.paused:
            state.status = "PAUSED"
            state.message = "Transfer paused by user"
            self.terminal.pump(state, force=True)
            self._apply_requests(state)
            if self.terminal.stop_requested:
                raise DownloadCancelled("Stopped by user; partial files are retained for resuming.")
            time.sleep(0.15)

    def _wait_for_space(self, remote_file: RemoteFile, partial_size: int, state: State) -> None:
        remaining = max(0, remote_file.size - partial_size)
        if remote_file.size > self.max_bytes:
            raise DownloadError(
                f"{remote_file.path} is {readable_size(remote_file.size)}, larger than the "
                f"{readable_size(self.max_bytes)} folder limit. Increase --max-gb or use another destination."
            )
        while True:
            self._check_controls(state)
            used = folder_size(self.destination)
            state.storage_used = used
            projected = used + remaining
            if used >= self.max_bytes:
                self.limit_was_reached = True
            can_resume_after_limit = not self.limit_was_reached or used < self.resume_bytes
            has_room = projected <= self.max_bytes
            if can_resume_after_limit and has_room:
                if self.limit_was_reached:
                    self.limit_was_reached = False
                state.waiting_reason = ""
                return
            state.status = "WAITING"
            if self.limit_was_reached and used >= self.resume_bytes:
                state.waiting_reason = (
                    f"Folder is {readable_size(used)}. Waiting until it is below {readable_size(self.resume_bytes)}."
                )
            else:
                state.waiting_reason = (
                    f"Need {readable_size(remaining)} free before this file; folder is {readable_size(used)}."
                )
            state.message = f"Checking again in {self.poll_seconds}s"
            deadline = time.monotonic() + self.poll_seconds
            while time.monotonic() < deadline:
                self._check_controls(state)
                time.sleep(0.15)

    def _wait_for_retry(self, seconds: int, state: State) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._check_controls(state)
            time.sleep(0.15)

    def _download_once(self, remote_file: RemoteFile, state: State) -> None:
        target = safe_target(self.destination, remote_file.path)
        partial = target.with_name(target.name + self.partial_suffix)
        try:
            partial_size = partial.stat().st_size
        except OSError:
            partial_size = 0
        if partial_size > remote_file.size:
            partial.unlink(missing_ok=True)
            partial_size = 0
        self._wait_for_space(remote_file, partial_size, state)
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HUB_URL}/{self.repository.url_prefix}{self.repository.repo_id}/resolve/{quote(self.repository.revision, safe='')}/{quote(remote_file.path, safe='/')}"
        headers = {"User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if partial_size:
            headers["Range"] = f"bytes={partial_size}-"
        try:
            response = urlopen(Request(url, headers=headers), timeout=SOCKET_TIMEOUT)
        except HTTPError as error:
            if error.code in RETRYABLE_HTTP_CODES:
                raise RetryableDownloadError(f"Hub returned HTTP {error.code}") from error
            raise DownloadError(f"Could not download {remote_file.path} ({error.code}).") from error
        except (URLError, OSError, ssl.SSLError) as error:
            reason = getattr(error, "reason", error)
            raise RetryableDownloadError(f"Network error: {reason}") from error
        with response:
            resumed = response.status == 206 and partial_size > 0
            if partial_size and not resumed:
                partial_size = 0
            mode = "ab" if resumed else "wb"
            state.status = "DOWNLOADING"
            state.message = "Streaming from Hugging Face"
            state.current_path = remote_file.path
            state.current_size = remote_file.size
            state.current_done = partial_size
            with partial.open(mode) as output:
                last_data = time.monotonic()
                while True:
                    self._check_controls(state)
                    # A half-dead connection (common right after a VPN switch) can
                    # trickle bytes forever: the socket timeout applies per recv, so
                    # it never fires. Tear the connection down and resume instead.
                    state.stalled_for = time.monotonic() - last_data
                    if state.stalled_for > STALL_SECONDS:
                        raise RetryableDownloadError(
                            f"No data for {state.stalled_for:.0f}s; dropping the stalled connection"
                        )
                    # Painted before the blocking read so the screen shows the wait
                    # rather than freezing on a stale frame.
                    self.terminal.pump(state, force=state.stalled_for > 3)
                    try:
                        block = response.read(CHUNK_SIZE)
                    except (IncompleteRead, OSError, ssl.SSLError) as error:
                        raise RetryableDownloadError(f"Connection interrupted: {error}") from error
                    if not block:
                        break
                    output.write(block)
                    last_data = time.monotonic()
                    state.stalled_for = 0.0
                    state.current_done += len(block)
                    state.storage_used += len(block)
        actual_size = partial.stat().st_size
        if actual_size != remote_file.size:
            raise RetryableDownloadError(
                f"Incomplete download for {remote_file.path}: got {readable_size(actual_size)}, "
                f"expected {readable_size(remote_file.size)}"
            )
        os.replace(partial, target)

    def _await_retry_reset(
        self, remote_file: RemoteFile, error: RetryableDownloadError | None, state: State
    ) -> bool:
        """Hold at the spent retry budget so the operator can grant more from the TUI."""

        if not self.terminal.enabled:
            return False
        self.terminal.reset_requested = False
        state.status = "EXHAUSTED"
        state.message = f"Retry budget spent on {remote_file.path}"
        # The call to action is drawn by the transfer screen for the EXHAUSTED status.
        state.waiting_reason = str(error)
        while True:
            self._check_controls(state)
            if self.terminal.reset_requested:
                self.terminal.reset_requested = False
                state.waiting_reason = ""
                state.status = "RETRYING"
                state.message = f"Retry budget reset to {self.retries}; resuming {remote_file.path}"
                self.terminal.pump(state, force=True)
                return True
            self.terminal.pump(state, force=True)
            time.sleep(0.15)

    def _download_file(self, remote_file: RemoteFile, state: State) -> None:
        last_error: RetryableDownloadError | None = None
        attempt = 0
        while True:
            attempt += 1
            state.retry_attempt = attempt
            try:
                self._download_once(remote_file, state)
                state.retry_attempt = 0
                return
            except RetryableDownloadError as error:
                last_error = error
                state.stalled_for = 0.0
                if attempt > self.retries:
                    if self._await_retry_reset(remote_file, last_error, state):
                        attempt = 0
                        continue
                    raise DownloadError(
                        f"Could not finish {remote_file.path} after {self.retries} retries: {last_error}. "
                        "Its .part file is retained; run the command again to resume."
                    ) from error
                delay = min(30, 2 ** (attempt - 1))
                state.status = "RETRYING"
                state.message = f"Connection interrupted; resuming attempt {attempt}/{self.retries} in {delay}s"
                state.waiting_reason = str(error)
                self.terminal.pump(state, force=True)
                self._wait_for_retry(delay, state)

    def run(self) -> None:
        self.destination.mkdir(parents=True, exist_ok=True)
        state = State(
            repository=self.repository.label,
            destination=str(self.destination.resolve()),
            storage_limit=self.max_bytes,
            resume_limit=self.resume_bytes,
            retry_limit=self.retries,
        )
        self.terminal.open()
        try:
            self.terminal.pump(state, force=True)
            files = list_files(self.repository, self.token, expand=self.settings.verify)
            if self.repository.subfolder:
                # The quant folder is usually self-contained, but the runtime
                # extras (vision projector, tokenizer config) often sit at the
                # repo root next to it. Fetch the whole manifest once to find
                # those without also pulling every other quant.
                whole = list_files(
                    Repository(
                        repo_id=self.repository.repo_id,
                        repo_type=self.repository.repo_type,
                        revision=self.repository.revision,
                    ),
                    self.token,
                    expand=self.settings.verify,
                )
                chosen = {item.path for item in files}
                for item in whole:
                    if item.path in chosen or not is_runtime_extra(item.path):
                        continue
                    files.append(item)
                    self.supporting.append(item.path)
            store = SessionStore(self.destination, self.repository)
            state.total_files = len(files)
            state.total_bytes = sum(item.size for item in files)
            state.storage_used = folder_size(self.destination)

            audit = self._reconcile(files, store, state)
            # Files sitting in the folder count as done without being refetched,
            # but they stay in the manifest so the final sweep still audits them.
            store.mark_many(audit.adopted)
            settled = audit.settled if not self.retry_missing else audit.adopted
            state.complete_files = len(settled)
            state.complete_bytes = sum(item.size for item in settled)
            state.reconciled = audit.summary()

            queue = list(audit.outstanding)
            if self.retry_missing:
                queue += audit.recorded
            queue.sort(key=lambda item: item.path)

            for remote_file in queue:
                self._check_controls(state)
                self._download_file(remote_file, state)
                store.mark_complete(remote_file)
                state.complete_files += 1
                state.complete_bytes += remote_file.size
                state.storage_used = folder_size(self.destination)

            self._finish(files, store, state, audit)
        finally:
            self.terminal.close()

    def _reconcile(self, files: list[RemoteFile], store: SessionStore, state: State) -> Audit:
        """Work out what the folder already holds before fetching anything."""

        state.status = "PREPARING"
        state.message = (
            "Checksumming files already in the folder…"
            if self.settings.verify
            else "Reconciling files already in the folder…"
        )
        self.terminal.pump(state, force=True)
        checked = 0

        def on_check(remote: RemoteFile) -> None:
            nonlocal checked
            checked += 1
            if self.settings.verify:
                state.current_path = remote.path
                state.current_size = remote.size
                state.current_done = 0
                state.message = f"Checksumming {checked:,} of {len(files):,} files already in the folder…"
            self._check_controls(state)

        def on_block(count: int) -> None:
            # Hashing a 40 GB shard takes minutes; keep the screen live and [q] usable.
            state.current_done += count
            self._check_controls(state)
            self.terminal.pump(state)

        audit = audit_destination(
            files,
            self.destination,
            store.completed,
            self.settings.verify,
            self.partial_suffix,
            on_check,
            on_block,
        )
        state.current_path = "—"
        state.current_size = 0
        state.current_done = 0
        return audit

    def _finish(self, files: list[RemoteFile], store: SessionStore, state: State, first: Audit) -> None:
        """Confirm every manifest entry is accounted for before declaring success."""

        state.message = "Verifying every repository file is accounted for…"
        self.terminal.pump(state, force=True)
        final = audit_destination(
            files, self.destination, store.completed, False, self.partial_suffix
        )
        state.current_path = "—"
        state.current_size = 0
        state.current_done = 0
        state.reconciled = final.summary()
        if final.outstanding:
            missing = ", ".join(item.path for item in final.outstanding[:3])
            more = f" (+{len(final.outstanding) - 3:,} more)" if len(final.outstanding) > 3 else ""
            raise DownloadError(
                f"{len(final.outstanding):,} of {len(files):,} files are still missing or the wrong "
                f"size: {missing}{more}. Run the command again to finish them."
            )
        state.status = "COMPLETE"
        adopted = len(first.adopted)
        moved = len(final.recorded)
        detail = []
        if adopted:
            verb = "was" if adopted == 1 else "were"
            detail.append(f"{adopted:,} already present {verb} {'verified' if self.settings.verify else 'kept'}")
        if moved:
            were = "is" if moved == 1 else "are"
            detail.append(f"{moved:,} completed earlier {were} no longer in the folder")
        if self.supporting:
            names = ", ".join(self.supporting)
            where = self.repository.subfolder
            detail.append(
                f"with {len(self.supporting):,} supporting file(s) from outside {where}: {names}"
            )
        state.message = "All {:,} repository files are accounted for.".format(len(files))
        if detail:
            state.message += "  " + "  ·  ".join(detail) + "."
        self.terminal.pump(state, force=True)
        if not self.terminal.enabled:
            print(f"Complete: {len(files):,} files, {readable_size(state.complete_bytes)}")
            if detail:
                print("  " + "; ".join(detail) + ".")
            if final.unknown:
                print(f"  {len(final.unknown):,} file(s) in the folder are not part of this repository.")
            if final.stale:
                print(
                    f"  {len(final.stale):,} partial file(s) from another revision are still using "
                    f"{readable_size(final.stale_bytes)}."
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable Hugging Face downloader with a folder-size gate. "
            "Everything is configurable inside the TUI; these flags only pre-fill it "
            "(and drive the run outright under --no-tui)."
        )
    )
    parser.add_argument(
        "repository",
        nargs="?",
        help="owner/repo or a huggingface.co URL (a ?show_file_info= link selects that quant's folder)",
    )
    parser.add_argument("--revision", help="Revision, branch, tag, or commit")
    parser.add_argument("--subfolder", help="Only download files under this subfolder (overrides the URL's)")
    parser.add_argument("--output", help="Parent folder for the model folder")
    parser.add_argument("--dir-name", help="Override the model-named subfolder")
    parser.add_argument("--no-subdir", action="store_true", help="Download straight into the parent folder")
    parser.add_argument("--max-gb", type=float, help="GB the download folder may never exceed")
    parser.add_argument("--resume-gb", type=float, help="Once full, wait until the folder is below this many GB")
    parser.add_argument("--poll-seconds", type=int, help="Seconds between storage checks (1-60)")
    parser.add_argument("--retries", type=int, help=f"Retries for a dropped connection (default: {DEFAULT_RETRIES})")
    parser.add_argument(
        "--token",
        help="Hugging Face access token (defaults to HF_TOKEN, then the saved token)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Checksum files already in the folder instead of trusting their size",
    )
    parser.add_argument("--retry-missing", action="store_true", help="Re-download files completed but since moved")
    parser.add_argument("--no-tui", action="store_true", help="Run headless with the values given here")
    return parser


def settings_from_args(args: argparse.Namespace, tokens: TokenStore) -> Settings:
    # Precedence: --token, then HF_TOKEN, then whatever was saved last time.
    settings = Settings(token=os.environ.get("HF_TOKEN", "") or tokens.load())
    if args.repository:
        settings.repository = args.repository
    if args.revision:
        settings.revision = args.revision
    if args.subfolder:
        settings.subfolder = args.subfolder
    if args.output:
        settings.output = args.output
    if args.dir_name:
        settings.dir_name = args.dir_name
    if args.no_subdir:
        settings.use_subdir = False
    if args.max_gb is not None:
        settings.max_gb = args.max_gb
    if args.resume_gb is not None:
        settings.resume_gb = args.resume_gb
    elif args.max_gb is not None:
        settings.resume_gb = round(args.max_gb * 0.6, 2)
    if args.poll_seconds is not None:
        settings.poll_seconds = args.poll_seconds
    if args.retries is not None:
        settings.retries = args.retries
    if args.token:
        settings.token = args.token
    if args.verify:
        settings.verify = True
    if args.retry_missing:
        settings.retry_missing = True
    settings.normalize()
    return settings


def main() -> int:
    args = build_parser().parse_args()
    tokens = TokenStore()
    settings = settings_from_args(args, tokens)
    terminal = Terminal(enabled=not args.no_tui)
    terminal.open()
    try:
        if terminal.enabled:
            entered = settings.token
            if not terminal.configure(settings):
                return 0
            if settings.token != entered:
                tokens.remember(settings.token)
        problem = settings.validate()
        if problem:
            raise DownloadError(problem)
        downloader = Downloader(settings, terminal)

        def stop_handler(_: int, __: Any) -> None:
            downloader.stop_requested = True

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        downloader.run()
        return 0
    except (DownloadError, DownloadCancelled) as error:
        terminal.close()
        print(f"\nStopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Reachable before the signal handlers are installed, i.e. from the panel.
        terminal.close()
        print("\nStopped: interrupted before the transfer started.", file=sys.stderr)
        return 130
    finally:
        terminal.close()


if __name__ == "__main__":
    raise SystemExit(main())
