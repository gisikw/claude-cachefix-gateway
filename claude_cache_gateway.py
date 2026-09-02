#!/usr/bin/env python3
"""Dependency-free Claude Code prompt-cache observer and repair proxy.

Point ANTHROPIC_BASE_URL at this process. It forwards credentials and bodies to
Anthropic unchanged in observe mode. Repair mode moves cache_control markers
off known request-time synthetic blocks and onto the latest stable content.

The JSONL log contains structure and token metrics, never prompts, responses,
credentials, or full headers.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import ssl
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO

VERSION = "0.1.0"
DEFAULT_UPSTREAM = "https://api.anthropic.com"
CONTINUATION_PROMPTS = {
    "<tool-result>Tool call complete. Results are above.</tool-result>",
    "Continue from where you left off.",
}
NO_RESPONSE = "No response requested."
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
MAX_USAGE_BUFFER = 4 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compact_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def text_of(block: Any) -> str:
    if isinstance(block, str):
        return block.strip()
    if isinstance(block, dict) and isinstance(block.get("text"), str):
        return block["text"].strip()
    return ""


def content_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def is_total_tokens(block: Any) -> bool:
    return text_of(block).startswith("<total_tokens>")


def is_continuation(block: Any) -> bool:
    return text_of(block) in CONTINUATION_PROMPTS


def is_no_response(block: Any) -> bool:
    return text_of(block) == NO_RESPONSE


def message_is_exact_text(message: Any, candidates: set[str]) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return content.strip() in candidates
    blocks = content_blocks(message)
    return len(blocks) == 1 and text_of(blocks[0]) in candidates


def stable_search_end(messages: list[Any], trailing_system_start: int) -> int:
    """Exclude a known no-op continuation pair immediately before reminders."""
    end = trailing_system_start
    if end >= 2:
        assistant = messages[end - 2]
        user = messages[end - 1]
        if (
            isinstance(assistant, dict)
            and assistant.get("role") == "assistant"
            and message_is_exact_text(assistant, {NO_RESPONSE})
            and isinstance(user, dict)
            and user.get("role") == "user"
            and message_is_exact_text(user, CONTINUATION_PROMPTS)
        ):
            return end - 2
    return end


def latest_stable_block(messages: list[Any], before: int) -> tuple[dict[str, Any] | None, str | None]:
    """Find the latest retained content block, ignoring synthetic text tails."""
    for mi in range(before - 1, -1, -1):
        message = messages[mi]
        blocks = content_blocks(message)
        for bi in range(len(blocks) - 1, -1, -1):
            block = blocks[bi]
            if is_total_tokens(block) or is_continuation(block) or is_no_response(block):
                continue
            role = message.get("role", "unknown") if isinstance(message, dict) else "unknown"
            return block, f"messages[{mi}].{role}[{bi}]"
    return None, None


def ordered_breakpoints(body: Any) -> list[dict[str, Any]]:
    """List cache breakpoints in Anthropic processing order without content."""
    if not isinstance(body, dict):
        return []
    result: list[dict[str, Any]] = []

    def add(block: Any, path: str) -> None:
        if not isinstance(block, dict) or "cache_control" not in block:
            return
        control = block.get("cache_control")
        if not isinstance(control, dict):
            return
        kind = "content"
        if is_total_tokens(block):
            kind = "total_tokens_reminder"
        elif is_continuation(block):
            kind = "continuation_prompt"
        elif is_no_response(block):
            kind = "no_response_sentinel"
        result.append(
            {
                "path": path,
                "kind": kind,
                "type": block.get("type"),
                "ttl": control.get("ttl", "5m"),
            }
        )

    for i, tool in enumerate(body.get("tools") or []):
        add(tool, f"tools[{i}]")
    system = body.get("system") or []
    if isinstance(system, list):
        for i, block in enumerate(system):
            add(block, f"system[{i}]")
    messages = body.get("messages") or []
    if isinstance(messages, list):
        for mi, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role", "unknown")
            for bi, block in enumerate(content_blocks(message)):
                add(block, f"messages[{mi}].{role}[{bi}]")
    return result


@dataclass
class RepairResult:
    body: bytes
    json_body: dict[str, Any] | None
    detected: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None


def analyze_and_maybe_repair(raw: bytes, repair: bool) -> RepairResult:
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return RepairResult(raw, None, parse_error=type(exc).__name__)
    if not isinstance(body, dict):
        return RepairResult(raw, None, parse_error="body_not_object")

    messages = body.get("messages")
    if not isinstance(messages, list):
        return RepairResult(raw, body)

    detected: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    trailing_start = len(messages)
    while trailing_start > 0:
        message = messages[trailing_start - 1]
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        trailing_start -= 1

    stable_end = stable_search_end(messages, trailing_start)
    target, target_path = latest_stable_block(messages, stable_end)

    candidates: list[tuple[dict[str, Any], str, str]] = []
    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown")
        for bi, block in enumerate(content_blocks(message)):
            if "cache_control" not in block or not isinstance(block.get("cache_control"), dict):
                continue
            path = f"messages[{mi}].{role}[{bi}]"
            if is_total_tokens(block) and mi >= trailing_start:
                candidates.append((block, path, "total_tokens_reminder"))
            elif is_continuation(block) and mi >= max(0, stable_end - 1):
                candidates.append((block, path, "continuation_prompt"))

    for block, path, kind in candidates:
        control = block.get("cache_control")
        event = {
            "kind": kind,
            "path": path,
            "ttl": control.get("ttl", "5m") if isinstance(control, dict) else None,
            "target": target_path,
        }
        detected.append(event)
        if not repair or target is None or control is None:
            continue
        # Process candidates in message order. A 5m continuation marker usually
        # arrives before a 1h reminder marker; preserving the first marker also
        # preserves Anthropic's required non-increasing TTL order.
        if not isinstance(target.get("cache_control"), dict):
            target["cache_control"] = control
            action = "moved"
        else:
            action = "removed_conflicting_marker"
        del block["cache_control"]
        repairs.append({**event, "action": action})

    output = compact_json(body) if repairs else raw
    return RepairResult(output, body, detected, repairs)


def find_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("usage"), dict):
        return value["usage"]
    for key in ("message", "response"):
        child = value.get(key)
        if isinstance(child, dict) and isinstance(child.get("usage"), dict):
            return child["usage"]
    return None


class UsageObserver:
    """Incrementally extracts usage from SSE or a bounded JSON response."""

    def __init__(self) -> None:
        self._line = bytearray()
        self._body = bytearray()
        self.usage: dict[str, Any] = {}
        self.cache_creation: dict[str, Any] = {}

    def feed(self, chunk: bytes) -> None:
        if len(self._body) < MAX_USAGE_BUFFER:
            room = MAX_USAGE_BUFFER - len(self._body)
            self._body.extend(chunk[:room])
        self._line.extend(chunk)
        while True:
            index = self._line.find(b"\n")
            if index < 0:
                if len(self._line) > 1024 * 1024:
                    self._line.clear()
                break
            line = bytes(self._line[:index]).rstrip(b"\r")
            del self._line[: index + 1]
            if line.startswith(b"data:"):
                data = line[5:].strip()
                if data and data != b"[DONE]":
                    self._observe_json(data)

    def finish(self) -> dict[str, Any]:
        self._observe_json(bytes(self._body))
        details = self.usage.get("output_tokens_details")
        reasoning = None
        if isinstance(details, dict):
            reasoning = details.get("thinking_tokens", details.get("reasoning_tokens"))
        completion_details = self.usage.get("completion_tokens_details")
        if reasoning is None and isinstance(completion_details, dict):
            reasoning = completion_details.get("reasoning_tokens")
        creation = self.cache_creation
        result = {
            "input_tokens": self.usage.get("input_tokens", self.usage.get("prompt_tokens")),
            "output_tokens": self.usage.get("output_tokens", self.usage.get("completion_tokens")),
            "cache_read_tokens": self.usage.get("cache_read_input_tokens"),
            "cache_write_tokens": self.usage.get("cache_creation_input_tokens"),
            "cache_write_5m_tokens": creation.get("ephemeral_5m_input_tokens"),
            "cache_write_1h_tokens": creation.get("ephemeral_1h_input_tokens"),
            "reasoning_tokens": reasoning,
        }
        # Preserve unknown as null, but retain explicitly reported zero.
        return result

    def _observe_json(self, data: bytes) -> None:
        try:
            value = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        usage = find_usage(value)
        if not usage:
            return
        for key, item in usage.items():
            if key == "cache_creation" and isinstance(item, dict):
                self.cache_creation.update(item)
            elif item is not None:
                self.usage[key] = item


class RunStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.successes = 0
        self.requests_with_unstable_markers = 0
        self.requests_repaired = 0
        self.suspected_full_rewrites = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0

    def add(self, record: dict[str, Any]) -> None:
        usage = record.get("usage") or {}
        with self._lock:
            self.requests += 1
            self.successes += record.get("outcome") == "ok"
            self.requests_with_unstable_markers += bool(record.get("unstable_breakpoints"))
            self.requests_repaired += bool(record.get("repairs"))
            self.suspected_full_rewrites += bool(record.get("suspected_full_cache_rewrite"))
            for attribute, key in (
                ("cache_read_tokens", "cache_read_tokens"),
                ("cache_write_tokens", "cache_write_tokens"),
                ("output_tokens", "output_tokens"),
            ):
                value = usage.get(key)
                if isinstance(value, int):
                    setattr(self, attribute, getattr(self, attribute) + value)

    def snapshot(self, mode: str) -> dict[str, Any]:
        with self._lock:
            denominator = self.cache_read_tokens + self.cache_write_tokens
            ratio = self.cache_read_tokens / denominator if denominator else None
            if self.suspected_full_rewrites >= 2:
                conclusion = "repeated_full_cache_rewrites_observed"
            elif self.requests_with_unstable_markers:
                conclusion = "known_unstable_breakpoints_observed"
            elif self.successes < 10:
                conclusion = "insufficient_data"
            else:
                conclusion = "no_known_marker_bug_observed"
            return {
                "type": "run_summary",
                "timestamp": utc_now(),
                "mode": mode,
                "requests": self.requests,
                "successes": self.successes,
                "requests_with_unstable_markers": self.requests_with_unstable_markers,
                "requests_repaired": self.requests_repaired,
                "suspected_full_cache_rewrites": self.suspected_full_rewrites,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "cache_read_ratio": round(ratio, 4) if ratio is not None else None,
                "output_tokens": self.output_tokens,
                "conclusion": conclusion,
            }


class JSONLLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file: BinaryIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.chmod(path, 0o600)
            self._file = os.fdopen(fd, "ab", buffering=0)

    def write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        line = compact_json(record) + b"\n"
        with self._lock:
            self._file.write(line)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


@dataclass
class Config:
    upstream: urllib.parse.SplitResult
    mode: str
    logger: JSONLLogger
    stats: RunStats
    quiet: bool
    timeout: float


class CacheGatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: Config):
        super().__init__(address, CacheGatewayHandler)
        self.config = config


class CacheGatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"claude-cache-gateway/{VERSION}"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/_cache_gateway/health":
            payload = compact_json({"ok": True, "version": VERSION, "mode": self.server.config.mode})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy()

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid BaseHTTPRequestHandler's potentially content-bearing request log.
        return

    @property
    def config(self) -> Config:
        return self.server.config  # type: ignore[attr-defined]

    def _proxy(self) -> None:
        started = time.monotonic()
        request_id = uuid.uuid4().hex[:16]
        sent_headers = False
        upstream_response: http.client.HTTPResponse | None = None
        connection: http.client.HTTPConnection | None = None
        status: int | None = None
        outcome = "proxy-error"
        error: str | None = None
        usage: dict[str, Any] = {}

        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header else 0
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        raw_body = self.rfile.read(length) if length else b""
        try:
            original_json = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            original_json = None
        result = analyze_and_maybe_repair(raw_body, self.config.mode == "repair")
        before = ordered_breakpoints(original_json)
        outbound_json = result.json_body
        if result.repairs:
            try:
                outbound_json = json.loads(result.body)
            except json.JSONDecodeError:
                outbound_json = None
        after = ordered_breakpoints(outbound_json)
        model = result.json_body.get("model") if isinstance(result.json_body, dict) else None
        session = self.headers.get("X-Claude-Code-Session-Id", "")
        session_hash = hashlib.sha256(session.encode()).hexdigest()[:12] if session else None

        try:
            connection = make_connection(self.config.upstream, self.config.timeout)
            target = upstream_target(self.config.upstream, self.path)
            headers = outbound_headers(self.headers, self.config.upstream)
            connection.request(self.command, target, body=result.body, headers=headers)
            upstream_response = connection.getresponse()
            status = upstream_response.status
            self.send_response_only(status, upstream_response.reason)
            for key, value in upstream_response.getheaders():
                lower = key.lower()
                if lower in HOP_BY_HOP or lower == "server":
                    continue
                self.send_header(key, value)
            # http.client removes upstream chunk framing. Closing the downstream
            # connection provides correct framing while preserving streaming.
            self.send_header("Connection", "close")
            self.end_headers()
            sent_headers = True
            self.close_connection = True

            observer = UsageObserver()
            while True:
                chunk = upstream_response.read(64 * 1024)
                if not chunk:
                    break
                observer.feed(chunk)
                if self.command != "HEAD":
                    self.wfile.write(chunk)
                    self.wfile.flush()
            usage = observer.finish()
            outcome = "ok" if 200 <= status < 300 else "upstream-error"
        except (BrokenPipeError, ConnectionResetError):
            outcome = "client-disconnect"
            error = "client_disconnected"
        except Exception as exc:  # keep the proxy alive, record only exception type
            error = type(exc).__name__
            if not sent_headers:
                payload = compact_json({"type": "error", "error": {"type": "proxy_error", "message": "cache gateway upstream failure"}})
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception:
                    pass
            if os.environ.get("CLAUDE_CACHE_GATEWAY_DEBUG") == "1":
                traceback.print_exc()
        finally:
            if upstream_response is not None:
                upstream_response.close()
            if connection is not None:
                connection.close()

        cache_write = usage.get("cache_write_tokens")
        cache_read = usage.get("cache_read_tokens")
        suspected_full_rewrite = bool(
            isinstance(cache_write, int)
            and cache_write >= 50_000
            and (not isinstance(cache_read, int) or cache_write > max(cache_read * 2, 50_000))
        )
        record = {
            "timestamp": utc_now(),
            "request_id": request_id,
            "mode": self.config.mode,
            "method": self.command,
            "path": urllib.parse.urlsplit(self.path).path,
            "model": model,
            "claude_session_hash": session_hash,
            "request_bytes": len(raw_body),
            "outbound_bytes": len(result.body),
            "breakpoints_before": before,
            "breakpoints_after": after,
            "unstable_breakpoints": result.detected,
            "repairs": result.repairs,
            "response_status": status,
            "outcome": outcome,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "usage": usage,
            "suspected_full_cache_rewrite": suspected_full_rewrite,
            "error": error,
        }
        self.config.logger.write(record)
        self.config.stats.add(record)
        if not self.config.quiet:
            action = "repaired" if result.repairs else ("detected" if result.detected else "clean")
            print(
                f"[cache-gateway] {status or '-'} model={model or '-'} cache={action} "
                f"read={cache_read if cache_read is not None else '-'} "
                f"write={cache_write if cache_write is not None else '-'} "
                f"output={usage.get('output_tokens') if usage.get('output_tokens') is not None else '-'} "
                f"full_rewrite={'yes' if suspected_full_rewrite else 'no'}",
                file=sys.stderr,
                flush=True,
            )


def make_connection(upstream: urllib.parse.SplitResult, timeout: float) -> http.client.HTTPConnection:
    host = upstream.hostname
    if not host:
        raise ValueError("upstream has no hostname")
    if upstream.scheme == "https":
        return http.client.HTTPSConnection(host, upstream.port or 443, timeout=timeout, context=ssl.create_default_context())
    if upstream.scheme == "http":
        return http.client.HTTPConnection(host, upstream.port or 80, timeout=timeout)
    raise ValueError(f"unsupported upstream scheme: {upstream.scheme}")


def upstream_target(upstream: urllib.parse.SplitResult, incoming: str) -> str:
    request = urllib.parse.urlsplit(incoming)
    base = upstream.path.rstrip("/")
    path = request.path if request.path.startswith("/") else "/" + request.path
    target = base + path
    if request.query:
        target += "?" + request.query
    return target


def outbound_headers(source: Any, upstream: urllib.parse.SplitResult) -> dict[str, str]:
    headers: dict[str, str] = {}
    connection_tokens = {item.strip().lower() for item in source.get("Connection", "").split(",") if item.strip()}
    for key, value in source.items():
        lower = key.lower()
        if lower in HOP_BY_HOP or lower in connection_tokens or lower in {"host", "content-length"}:
            continue
        headers[key] = value
    host = upstream.hostname or ""
    default_port = 443 if upstream.scheme == "https" else 80
    if upstream.port and upstream.port != default_port:
        host += f":{upstream.port}"
    headers["Host"] = host
    headers["Via"] = f"1.1 claude-cache-gateway/{VERSION}"
    return headers


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("observe", "repair"), default="observe", help="observe leaves request bodies byte-identical; repair moves known unstable cache markers")
    parser.add_argument("--listen", default="127.0.0.1", help="listen address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM, help=f"upstream base URL (default: {DEFAULT_UPSTREAM})")
    parser.add_argument("--log", type=Path, default=Path("claude-cache-gateway.jsonl"), help="privacy-safe JSONL log path; use '-' to disable")
    parser.add_argument("--quiet", action="store_true", help="suppress one-line request summaries")
    parser.add_argument("--timeout", type=float, default=900.0, help="upstream socket timeout in seconds")
    parser.add_argument("--allow-remote", action="store_true", help="allow a non-loopback listen address")
    parser.add_argument("--allow-insecure-upstream", action="store_true", help="allow an http:// upstream (for local testing only)")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    upstream = urllib.parse.urlsplit(args.upstream)
    if upstream.scheme != "https" and not args.allow_insecure_upstream:
        print("error: upstream must be https:// (or pass --allow-insecure-upstream for a local test server)", file=sys.stderr)
        return 2
    loopbacks = {"127.0.0.1", "::1", "localhost"}
    if args.listen not in loopbacks and not args.allow_remote:
        print("error: refusing non-loopback listener without --allow-remote", file=sys.stderr)
        return 2
    log_path = None if str(args.log) == "-" else args.log
    logger = JSONLLogger(log_path)
    stats = RunStats()
    config = Config(upstream=upstream, mode=args.mode, logger=logger, stats=stats, quiet=args.quiet, timeout=args.timeout)
    server = CacheGatewayServer((args.listen, args.port), config)
    actual_host, actual_port = server.server_address[:2]

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"claude-cache-gateway {VERSION} mode={args.mode} listening on http://{actual_host}:{actual_port}", file=sys.stderr)
    print(f"upstream={args.upstream} log={log_path or 'disabled'}", file=sys.stderr)
    print(f"run Claude Code in another shell with: ANTHROPIC_BASE_URL=http://127.0.0.1:{actual_port} claude", file=sys.stderr)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        summary = stats.snapshot(args.mode)
        logger.write(summary)
        logger.close()
        print("[cache-gateway] run summary: " + json.dumps(summary, separators=(",", ":")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
