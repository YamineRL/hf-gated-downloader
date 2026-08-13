# hf-gated-downloader

**Download a Hugging Face model that is bigger than your disk.**

A single-file, dependency-free Python program that pulls a Hugging Face
repository while holding the destination folder under a size limit you set.
When the folder fills up, the download *waits* — you move finished shards
somewhere else, and it carries on where it left off.

[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)

<!-- Add a terminal screenshot or asciinema cast here. -->

---

## The problem

You want a 1.6 TB model. You have a 1 TB drive, or a 200 GB VPS, or a cache
partition you are not allowed to blow up. Every mainstream downloader takes the
same view: start, run until the disk is full, fail — and leave you to work out
which of the 300 shards actually finished.

`hf-gated-downloader` treats free space as a first-class constraint. You give it
a ceiling; it never crosses it.

```
Max folder size:  100 GB   ← the folder may never exceed this
Resume below:      60 GB   ← once full, wait until it drops under this
```

The loop that results:

1. Files download until the folder reaches 100 GB.
2. The downloader pauses and tells you it is waiting for space.
3. You move completed shards to another drive, an external disk, network
   storage — anywhere.
4. As soon as the folder drops below 60 GB, downloading resumes automatically.
5. Repeat until the repository is complete.

Moving files away does not confuse it. Completed paths are recorded in a session
store **outside** the destination folder, so a shard you relocated in step 3 is
still known to be done and is never fetched twice.

## Quick start

No install, no virtualenv, no `pip`. One file, standard library only.

```bash
curl -LO https://raw.githubusercontent.com/YamineRL/hf-gated-downloader/main/hf_gated_downloader.py
chmod +x hf_gated_downloader.py
./hf_gated_downloader.py
```

That opens the control panel. Set the repository, point it at a folder, set your
ceiling, press `s`.

Prefer flags? They pre-fill the same panel, or drive the run outright with
`--no-tui`:

```bash
# Interactive, pre-filled
./hf_gated_downloader.py meta-llama/Llama-3.3-70B-Instruct --max-gb 150

# Headless — for tmux, ssh, cron, a container
./hf_gated_downloader.py unsloth/DeepSeek-V4-GGUF \
    --output /mnt/models --max-gb 100 --resume-gb 60 --no-tui

# Gated or private repo
export HF_TOKEN=hf_xxxxxxxxxxxx
./hf_gated_downloader.py owner/private-repo

# Just one folder of a large repo
./hf_gated_downloader.py owner/repo --subfolder UD-Q4_K_XL --revision main
```

## Why not just use `hf download`?

Honest answer first: **if the model fits on your disk and you have a good
connection, use `hf download`.** It is faster, it is maintained by Hugging Face,
and it does things this script does not. This tool exists for the cases where
that is not enough.

| | `hf-gated-downloader` | `hf download` / `huggingface_hub` |
|---|---|---|
| **Folder-size ceiling** | Hard cap, enforced before every file | None — runs until the disk fills |
| **Behaviour when full** | Waits, polls for space, resumes by itself | Fails, often mid-shard |
| **Install** | Copy one `.py` file | `pip install huggingface_hub` + deps |
| **Runtime deps** | Python standard library only | `requests`, `tqdm`, `filelock`, `fsspec`, … |
| **Progress display** | Full control panel: throughput chart, per-file state, live settings | Progress bars |
| **Change settings mid-run** | Yes — retries, ceiling, poll interval, without restarting | Restart the command |
| **Files on disk** | Plain files in a plain folder | Blob store + symlinks in the HF cache |
| **Parallel downloads** | No — one file, one connection | Yes, and Xet chunk transfer |
| **Xet dedup transport** | No | Yes, where the repo supports it |
| **Python API / uploads** | No, CLI only | Yes |

### Choose this when

- **The repository is larger than the target disk.** This is the headline
  feature and nothing else in the ecosystem offers it. A drip-feed workflow —
  download 100 GB, move it off, download the next 100 GB — turns a 1.6 TB
  repository into something a 1 TB drive can fetch overnight.
- **The disk is shared and must not fill.** A ceiling is a safety rail on a box
  where something else needs the space: a build server, a workstation with a
  running database, a VPS where a full `/` takes the machine down.
- **You cannot install packages.** A locked-down server, a minimal container, a
  rescue shell, a machine with no working `pip`. `curl` the file and run it.
- **The connection is bad.** See [Reliability](#reliability) — resume from
  partial files, stall detection, backoff, and checksum verification are the
  parts of this script that got the most care.
- **You want plain files, not a cache.** Output mirrors the repository layout
  exactly — `folder/model-00001-of-00030.safetensors` and friends. No blob
  store, no symlinks to resolve before pointing llama.cpp or vLLM at it.
- **You want to watch it.** The panel shows current rate, a rolling throughput
  chart, folder usage against the ceiling, per-file status, and what it is
  waiting on — useful when a transfer runs for hours and you need to know
  whether it is progressing or quietly stuck.

### Choose `hf download` when

- Raw speed matters most and the model fits. Parallel file transfers and Xet
  win, and this script does not attempt to compete there.
- You want the shared HF cache, deduplicated across repos and tools.
- You need a Python API, uploads, or anything beyond fetching a repo.

## Reliability

Long downloads over imperfect links fail in specific ways. Each of these is
handled deliberately:

- **Resume from partial files.** Interrupted files are kept as `.part` and
  continued with an HTTP `Range` request. If the server answers `200` instead of
  `206`, the partial is discarded rather than appended to at a wrong offset — a
  subtle way to build a corrupt file that this avoids.
- **Revision-keyed partials.** The `.part` suffix embeds a hash of the
  repository revision, so a stale partial from a different revision can never be
  resumed into a file it does not belong to.
- **Stall detection.** A socket timeout only fires on a fully idle read, so a
  half-dead connection that trickles a byte a minute can hang forever. Time
  since last data is tracked separately; after 90 seconds the connection is torn
  down and retried.
- **Exponential backoff** on `408, 425, 429, 500, 502, 503, 504`, capped at 30
  seconds, 8 attempts by default and adjustable mid-run.
- **Checksum verification.** Files are checked against the LFS SHA-256 where
  the Hub exposes one, falling back to the git blob SHA-1 (framed correctly as
  `blob <len>\0` + contents).
- **Nothing truncated is ever published.** Size is checked before the atomic
  `os.replace` into the final path, so a partial transfer cannot masquerade as a
  finished file.
- **Path traversal is blocked.** Repository paths that are absolute or contain
  `..` are refused rather than written outside the destination.

## The control panel

Everything is configurable from the panel — no flag is ever required.

**Settings screen**

| Key | Action |
|---|---|
| `↑` `↓` / `j` `k` | Move between fields |
| `Enter` | Edit the selected field |
| `+` / `-` | Adjust numeric fields |
| `s` | Start the download |
| `q` | Quit |

**While downloading**

| Key | Action |
|---|---|
| `p` / `Space` | Pause / resume |
| `c` | Reopen settings — the ceiling, poll interval and retries apply live |
| `+` / `-` | Add or remove retries for the current file |
| `r` | Reset the current file and start it over |
| `q` | Stop cleanly, keeping partials for the next run |

`SIGINT` and `SIGTERM` stop cleanly too; `.part` files survive, and rerunning
the same command picks up where it stopped.

## Options

| Flag | Meaning |
|---|---|
| `repository` | `owner/repo`, or any `huggingface.co` URL |
| `--revision` | Branch, tag, or commit |
| `--subfolder` | Fetch only files under this path |
| `--output` | Parent folder the model folder is created in |
| `--dir-name` | Override the model-named subfolder |
| `--no-subdir` | Download straight into the parent folder |
| `--max-gb` | GB the folder may never exceed |
| `--resume-gb` | Once full, wait until the folder is below this |
| `--poll-seconds` | Seconds between storage checks (1–60) |
| `--retries` | Attempts after a dropped connection (default 8) |
| `--token` | Access token; defaults to `$HF_TOKEN` |
| `--verify` | Checksum files already in the folder instead of trusting size |
| `--retry-missing` | Re-fetch files completed earlier but since moved away |
| `--no-tui` | Run headless with the values given |

### Where state lives

| What | Where |
|---|---|
| Completed-file record | `$XDG_STATE_HOME/hf-gated-downloader/<hash>.json` (default `~/.local/state/…`) |
| In-flight files | `<name>.<revision-hash>.part`, next to the final file |

The session store is keyed by destination *and* repository, so several
downloads can run against different folders without interfering. Because it
sits outside the destination, files you move away stay recorded as complete —
that is what makes the drip-feed workflow possible. To re-fetch files you have
relocated, pass `--retry-missing`.

## Requirements

- Python 3.9 or newer, standard library only. Developed against 3.14.
- A terminal with `curses` — Linux, macOS, BSD, or WSL on Windows. Native
  Windows has no `curses`; run with `--no-tui`.

## Contributing

Issues and pull requests are welcome. Read [ROADMAP.md](ROADMAP.md) first: it
lists the known gaps (no parallelism, no Xet support, verification defaults) and
also the parts that look naive but are deliberate, so a well-meant "fix" does
not remove something load-bearing.

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

Not affiliated with or endorsed by Hugging Face.
