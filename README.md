# Claude Cache Gateway

A single-file, dependency-free pass-through gateway for diagnosing and repairing
Claude Code prompt-cache invalidation.

It is designed for a coworker to download one Python file, run it locally, and
point Claude Code at it with `ANTHROPIC_BASE_URL`. The gateway does **not** ask
for, load, or store credentials. It forwards the authentication headers Claude
Code already sends to Anthropic.

## What it detects

Some Claude Code request shapes put `cache_control` on request-time synthetic
content:

- the trailing role-system `<total_tokens>` reminder;
- a synthetic continuation prompt such as `Continue from where you left off.`;
- the equivalent tool-result continuation sentinel.

Those blocks can disappear and be regenerated between requests. A cache prefix
ending on one is then not reusable by the succeeding turn. The observable
signature is a cache-read count pinned near the fixed system prompt while
cache-creation grows toward the entire conversation size.

## Requirements

- Python 3.10 or newer.
- No pip packages.
- macOS, Linux, or Windows with Python installed.

Modern macOS does not guarantee a system Python. If `python3 --version` fails,
install Python with your normal developer tooling, for example:

```sh
brew install python
```

## Observe without changing requests

Terminal 1:

```sh
python3 claude_cache_gateway.py --mode observe
```

Terminal 2:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

`observe` mode forwards request bodies byte-for-byte. It logs whether unstable
breakpoints were present and extracts response usage from both buffered JSON
and SSE streams.

Claude Code reads `ANTHROPIC_BASE_URL` when it starts. Start a new `claude`
process after setting the variable.

## Repair known unstable markers

```sh
python3 claude_cache_gateway.py --mode repair
```

Then start Claude Code as above. Repair mode moves unstable cache markers onto
the latest stable real content block. It does not remove prompts, tool results,
messages, thinking signatures, or credentials.

Start in `observe`, reproduce the behavior, then repeat in `repair` for an A/B
comparison.

## What is logged

The default log is `claude-cache-gateway.jsonl`, created mode 0600. Each line
contains:

- timestamp and random request ID;
- model, API path, response status, outcome, and latency;
- a one-way hash of the Claude Code session ID when present;
- cache breakpoint paths, kinds, and TTLs before and after repair;
- input, output, cache-read, cache-write, and reasoning token counts when the
  upstream reports them;
- 5-minute versus 1-hour cache creation when reported;
- a conservative `suspected_full_cache_rewrite` flag.

It deliberately does **not** log:

- prompts or response text;
- tool inputs or tool results;
- request or response bodies;
- authorization, API-key, cookie, or full header values;
- raw session IDs.

Disable the file log with `--log -`. One-line summaries still appear on stderr
unless `--quiet` is supplied.

Example summary:

```text
[cache-gateway] 200 model=claude-opus-5 cache=detected read=15119 write=199545 output=358 full_rewrite=yes
```

## Recommended validation protocol

1. Start the gateway in `observe` mode.
2. Start a fresh Claude Code process through it.
3. Work normally for at least 20 requests in one non-compacted session. Cold
   starts alone are not evidence of a cache bug.
4. Include several tool turns and, if safe, one resume/continuation.
5. Keep request spacing below five minutes while testing so normal 5-minute
   expiry is not confused with invalidation.
6. Stop Claude Code and the gateway.
7. Repeat the same general workload in `repair` mode.
8. Compare cache-read/cache-write trajectories, not output length alone.

A healthy long session should generally read an increasingly large retained
prefix and write only the new tail. A suspicious session repeatedly writes most
of its growing context while reading only a small fixed prefix.

## Options

```text
--mode observe|repair
--listen 127.0.0.1
--port 8787
--upstream https://api.anthropic.com
--log claude-cache-gateway.jsonl
--quiet
--timeout 900
```

The listener is loopback-only by default. A non-loopback listener is refused
unless `--allow-remote` is explicit. HTTPS upstreams are required unless
`--allow-insecure-upstream` is supplied for local testing.

Health check:

```sh
curl http://127.0.0.1:8787/_cache_gateway/health
```

## Tests

```sh
python3 -m unittest -v
```

The distributable runtime remains the single file
`claude_cache_gateway.py`; the README and test file are project support.

## Scope and caveats

This is a narrow diagnostic proxy, not a general production gateway. It
supports ordinary Claude Code HTTP/1.1 Messages API traffic, including SSE. It
forwards credentials but never terminates or refreshes authentication itself.

Anthropic does not publish the exact subscription quota weighting formula.
Cache token telemetry can prove cache behavior; it cannot by itself prove the
precise amount charged against a Pro or Max allowance.
