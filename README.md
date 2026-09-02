# Claude Cache Gateway

A dependency-free, single-file pass-through gateway for observing and repairing
Claude Code prompt-cache requests.

It forwards Claude Code's existing credentials and streams Anthropic's response
back unchanged. It does not load, supply, store, or refresh credentials, and it
does not log prompts or response text.

## Precisely what it repairs

Only two request properties:

1. **TTL ordering.** Anthropic processes cache breakpoints in tools → system →
   messages order. Once a default/explicit 5-minute breakpoint appears, a later
   1-hour breakpoint is invalid. Repair mode downgrades the later marker to 5m.
2. **Trailing reminder marker.** If a trailing role-system `<total_tokens>`
   reminder owns a cache marker, repair mode moves that marker to the nearest
   preceding content block in an ordinary user or assistant turn. If that turn
   already has a marker, the existing marker wins and the trailing marker is
   removed.

It does **not** recognize, strip, relocate, or otherwise special-case tool
continuation messages. It does not perform projection cleanup.

## Run

Requires Python 3.10+ and no pip packages.

Observe without changing request bodies:

```sh
python3 claude_cache_gateway.py --mode observe
```

Repair the two cache conditions above:

```sh
python3 claude_cache_gateway.py --mode repair
```

Then start a new Claude Code process in another terminal:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

Claude Code reads `ANTHROPIC_BASE_URL` at process startup.

Modern macOS does not guarantee Python is installed. If `python3 --version`
fails, use the team's normal Python setup or, with Homebrew:

```sh
brew install python
```

## Per-request output

The gateway waits for Anthropic's response usage event and prints one line per
request:

```text
[cache-gateway] status=200 model=claude-opus-5 input=2 output=358 cache_read=194749 cache_write=19529 cache_write_5m=0 cache_write_1h=19529 reasoning=41 ordering_needed=1 ordering_repaired=1 tail_needed=1 tail_repaired=1
```

Fields unavailable from the upstream are printed as `-`.

- `ordering_needed`: the incoming request had invalid TTL ordering.
- `ordering_repaired`: repair mode changed at least one late 1h marker to 5m.
- `tail_needed`: a trailing `<total_tokens>` marker was present.
- `tail_repaired`: repair mode moved or removed that marker.

In observe mode, `*_needed` can be 1 while `*_repaired` remains 0.

There is no Ctrl-C aggregate report.

## JSONL log

The same request-level information is written to
`claude-cache-gateway.jsonl`, mode 0600. It includes structural breakpoint paths,
TTL changes, usage, status, and latency. It does not include:

- prompts or response text;
- tool inputs or results;
- request or response bodies;
- credentials, cookies, or complete headers;
- raw Claude Code session IDs.

Disable the file with `--log -`. Suppress terminal lines with `--quiet`.

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

The listener is loopback-only unless `--allow-remote` is explicit. HTTPS
upstreams are required unless `--allow-insecure-upstream` is supplied for local
testing.

Health check:

```sh
curl http://127.0.0.1:8787/_cache_gateway/health
```

## Tests

```sh
python3 -m unittest -v
```

The runtime distribution is the single file `claude_cache_gateway.py`; the
README and tests are support material. MIT licensed.
