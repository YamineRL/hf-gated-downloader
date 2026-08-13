# Roadmap

Known gaps and planned work, from an audit of the script on 2026-08-13. Read
this before opening a PR — some of what looks like low-hanging fruit is
deliberate, and the parts that genuinely need work are listed with the reasons
they have not been done yet.

## What is already good (do not "fix" these)

Worth stating, because a superficial read makes this look like a naive
`urllib` fetch loop. It isn't:

- **Range-based resume with 206 handling** (`~1341`). Sends
  `Range: bytes=<partial>-`, and if the server answers 200 instead of 206 it
  correctly discards `partial_size` and restarts the file rather than
  appending to a wrong offset.
- **Stall detection** (`STALL_SECONDS = 90`, `~1377`). A socket timeout is
  per-`recv`, so a half-dead connection that trickles a byte occasionally never
  trips it. This tracks time-since-last-data and tears the connection down.
  That is a real failure mode, correctly handled.
- **Exponential backoff**, `min(30, 2 ** (attempt - 1))` (`1452`), over a
  sensible retryable set: `{408, 425, 429, 500, 502, 503, 504}` (`33`).
- **Digest verification** (`digest_matches:375`). Prefers the LFS sha256, falls
  back to the git blob sha1 — and hashes `b"blob %d\0"` + contents, which is
  the actual git object framing. Easy to get wrong; this gets it right.
- **Revision-keyed partial files** (`~1234`). The `.part` suffix embeds a hash
  of the repo label, so a stale partial from a different revision cannot be
  resumed into a file it does not belong to.
- **Post-download size check** before `os.replace` (`~1405`), so a truncated
  transfer never lands at the final path.
- **Folder-size limit** — the feature the script was written for, and something
  no mainstream HF client offers.

## 1. No concurrency at all — the one real gap

`grep -cE 'import threading|Thread\(|asyncio|multiprocessing'` returns **0**.
One file at a time, one connection, one 1 MiB `response.read()` loop.

This is the entire reason `hf download` is faster; it parallelizes across
files. On a fast link a single HTTPS stream is the bottleneck long before the
line is.

Suggested, in increasing order of effort:

- **File-level parallelism.** A `ThreadPoolExecutor` over the manifest with
  `--jobs N` (4–8 is the useful range). Requires making `State` updates
  thread-safe and giving the TUI a per-worker row. Biggest win for the least
  structural change, since the transfer loop itself already works per-file.
- **Chunk-level parallelism within one file.** Split each shard into N ranges
  and fetch concurrently into one preallocated file. Helps when a repo is a few
  enormous shards rather than many files — which is exactly the Kimi/DeepSeek
  shape. More work: needs sparse writes and per-range resume bookkeeping.
- **Connection reuse.** Each file currently pays a fresh TCP + TLS handshake.
  `urllib` has no pooling. Even single-threaded, keep-alive across files is a
  measurable win on repos with many small files.

Caveat worth measuring before building: at some point the 1 MiB-per-block
Python loop becomes GIL-bound. Threads help because the work is I/O-wait, but
verify with a real repo rather than assuming.

## 2. Verification is off by default, and the digest fetch is coupled to it

`Settings.verify` defaults to `False` (`501`), and the manifest call is:

    files = list_files(self.repository, self.token, expand=self.settings.verify)   # 1471

`expand=true` is what makes the Hub return digests at all. So with the default
settings the script **never fetches a checksum**, and correctness rests
entirely on the size comparison at `~1405`.

Size-only is weaker than it looks. It catches truncation — the common case, and
the one colibri's own docs warn about ("a download can finish with a truncated
shard even when the client reports success", `docs/deepseek-v4.md`). It does not
catch silent corruption: a flipped byte mid-shard, or a proxy/CDN serving a
stale or wrong-but-same-length object.

Two independent changes, and they should not be conflated:

- **Decouple the digest fetch from the verify flag.** Always request
  `expand=true`. The comment at `291` notes it pages more slowly, so if that
  cost is real, fetch it lazily for files above some size rather than tying it
  to a user-facing integrity toggle.
- **Verify newly downloaded files by default.** `verify` currently means
  "checksum files already in the folder" (`572`) — an audit of pre-existing
  data. There is no separate control for hashing what this run just wrote.
  Hashing a shard costs one sequential read of data still in page cache;
  against a multi-hour download that is noise. Consider `--verify=none|new|all`
  with `new` as the default, keeping today's behaviour available as `none`.

## 3. No Xet support

`grep -c xet` returns **0**. Hugging Face has been migrating large repos to Xet,
its chunk-level dedup transport; `huggingface_hub` uses it automatically where
available. This script always takes the plain LFS path.

Consequences: no dedup benefit when re-pulling a revision that shares chunks
with one already on disk, and none of the parallel chunk fetching Xet gives for
free. Implementing the protocol from scratch is a large job and probably not
worth it for a single-file script — but it is the structural reason this will
stay slower than `hf download` on Xet-backed repos even after item 1. Worth a
line in `--help` so the tradeoff is explicit rather than surprising.

## Not recommended

- **Shelling out to `aria2c`/`hf_transfer`.** It would be faster, but it throws
  away the two things that make this script worth keeping: the folder-size limit
  and the curses control panel. If raw speed is the goal, `hf download` already
  exists — the reason to run this is the size cap.
- **Rewriting on `requests`/`httpx`.** Pooling and HTTP/2 are attractive, but
  the current code is dependency-free and single-file, which is why it works on
  a bare box. Item 1 is achievable within the stdlib.

## Where this came from

The script was written to pull a ~1.6 TB model repository onto a 1 TB drive by
never letting the download folder exceed a set size — shards get moved off to
other storage as they land, and the downloader waits for the space. That
constraint is why the size gate, the out-of-folder session store, and the
aggressive resume logic exist, and it is the use case to keep working when
changing any of them.
