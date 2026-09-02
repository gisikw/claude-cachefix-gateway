#!/usr/bin/env python3
"""Dependency-free Claude Code prompt-cache observer and repair proxy.

Point ANTHROPIC_BASE_URL at this process. It forwards credentials and bodies to
Anthropic unchanged in observe mode. Repair mode normalizes mixed cache TTL
ordering and moves a marker off a trailing <total_tokens> reminder onto the
nearest preceding ordinary user/assistant content block.

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
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO

VERSION = "0.2.1"
DEFAULT_UPSTREAM = "https://api.anthropic.com"
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


def ordered_cache_blocks(body: Any) -> list[tuple[dict[str, Any], str]]:
    """Return cache-bearing blocks in Anthropic's tools/system/messages order."""
    if not isinstance(body, dict):
        return []
    result: list[tuple[dict[str, Any], str]] = []

    def add(block: Any, path: str) -> None:
        if isinstance(block, dict) and isinstance(block.get("cache_control"), dict):
            result.append((block, path))

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


def ordered_breakpoints(body: Any) -> list[dict[str, Any]]:
    result = []
    for block, path in ordered_cache_blocks(body):
        control = block["cache_control"]
        result.append(
            {
                "path": path,
                "kind": "total_tokens_reminder" if is_total_tokens(block) else "content",
                "type": block.get("type"),
                "ttl": control.get("ttl", "5m"),
            }
        )
    return result


def ttl_order_issues(body: Any) -> list[dict[str, Any]]:
    """Find a 1h breakpoint occurring after a default/explicit 5m breakpoint."""
    short_seen = False
    issues = []
    for _block, path in ordered_cache_blocks(body):
        control = _block["cache_control"]
        ttl = control.get("ttl", "5m")
        if ttl == "1h" and short_seen:
            issues.append({"path": path, "from": "1h", "to": "5m"})
        elif ttl in (None, "", "5m"):
            short_seen = True
    return issues


def normalize_ttl_order(body: Any) -> list[dict[str, Any]]:
    repairs = []
    short_seen = False
    for block, path in ordered_cache_blocks(body):
        control = block["cache_control"]
        ttl = control.get("ttl", "5m")
        if ttl == "1h" and short_seen:
            control["ttl"] = "5m"
            repairs.append({"path": path, "from": "1h", "to": "5m"})
        elif ttl in (None, "", "5m"):
            short_seen = True
    return repairs


def latest_ordinary_turn_block(messages: list[Any], before: int) -> tuple[dict[str, Any] | None, str | None]:
    """Find the nearest preceding block in an ordinary user/assistant turn."""
    for mi in range(before - 1, -1, -1):
        message = messages[mi]
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        blocks = content_blocks(message)
        for bi in range(len(blocks) - 1, -1, -1):
            role = message["role"]
            return blocks[bi], f"messages[{mi}].{role}[{bi}]"
    return None, None


@dataclass
class RepairResult:
    body: bytes
    json_body: dict[str, Any] | None
    ordering_issues: list[dict[str, Any]] = field(default_factory=list)
    ordering_repairs: list[dict[str, Any]] = field(default_factory=list)
    tail_issues: list[dict[str, Any]] = field(default_factory=list)
    tail_repairs: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None


def analyze_and_maybe_repair(raw: bytes, repair: bool) -> RepairResult:
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return RepairResult(raw, None, parse_error=type(exc).__name__)
    if not isinstance(body, dict):
        return RepairResult(raw, None, parse_error="body_not_object")

    ordering_issues = ttl_order_issues(body)
    ordering_repairs = normalize_ttl_order(body) if repair else []
    tail_issues: list[dict[str, Any]] = []
    tail_repairs: list[dict[str, Any]] = []

    messages = body.get("messages")
    if isinstance(messages, list):
        trailing_start = len(messages)
        while trailing_start > 0:
            message = messages[trailing_start - 1]
            if not isinstance(message, dict) or message.get("role") != "system":
                break
            trailing_start -= 1
        target, target_path = latest_ordinary_turn_block(messages, trailing_start)
        for mi in range(trailing_start, len(messages)):
            message = messages[mi]
            for bi, block in enumerate(content_blocks(message)):
                control = block.get("cache_control")
                if not is_total_tokens(block) or not isinstance(control, dict):
                    continue
                event = {
                    "path": f"messages[{mi}].system[{bi}]",
                    "ttl": control.get("ttl", "5m"),
                    "target": target_path,
                }
                tail_issues.append(event)
                if repair and target is not None:
                    if not isinstance(target.get("cache_control"), dict):
                        target["cache_control"] = control
                        action = "moved"
                    else:
                        action = "removed_conflicting_marker"
                    del block["cache_control"]
                    tail_repairs.append({**event, "action": action})

    changed = bool(ordering_repairs or tail_repairs)
    output = compact_json(body) if changed else raw
    return RepairResult(output, body, ordering_issues, ordering_repairs, tail_issues, tail_repairs)


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
    """Incrementally extracts usage from compressed SSE or bounded JSON."""

    def __init__(self, content_encoding: str = "") -> None:
        self._line = bytearray()
        self._body = bytearray()
        self.usage: dict[str, Any] = {}
        self.cache_creation: dict[str, Any] = {}
        encoding = content_encoding.strip().lower()
        self._decoder: zlib.Decompress | None = None
        if encoding in {"gzip", "x-gzip"}:
            self._decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            self._decoder = zlib.decompressobj()

    def feed(self, chunk: bytes) -> None:
        if self._decoder is not None:
            try:
                chunk = self._decoder.decompress(chunk)
            except zlib.error:
                # Telemetry is observational: decoding failure must never
                # interrupt forwarding the original response to Claude Code.
                self._decoder = None
                return
        self._feed_decoded(chunk)

    def _feed_decoded(self, chunk: bytes) -> None:
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
        if self._decoder is not None:
            try:
                self._feed_decoded(self._decoder.flush())
            except zlib.error:
                pass
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
        if result.ordering_repairs or result.tail_repairs:
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

            observer = UsageObserver(upstream_response.getheader("Content-Encoding", ""))
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
            "ordering_issues": result.ordering_issues,
            "ordering_repairs": result.ordering_repairs,
            "tail_issues": result.tail_issues,
            "tail_repairs": result.tail_repairs,
            "response_status": status,
            "outcome": outcome,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "usage": usage,
            "error": error,
        }
        self.config.logger.write(record)
        if not self.config.quiet:
            print(
                f"[cache-gateway] status={status or '-'} model={model or '-'} "
                f"input={usage.get('input_tokens') if usage.get('input_tokens') is not None else '-'} "
                f"output={usage.get('output_tokens') if usage.get('output_tokens') is not None else '-'} "
                f"cache_read={cache_read if cache_read is not None else '-'} "
                f"cache_write={cache_write if cache_write is not None else '-'} "
                f"cache_write_5m={usage.get('cache_write_5m_tokens') if usage.get('cache_write_5m_tokens') is not None else '-'} "
                f"cache_write_1h={usage.get('cache_write_1h_tokens') if usage.get('cache_write_1h_tokens') is not None else '-'} "
                f"reasoning={usage.get('reasoning_tokens') if usage.get('reasoning_tokens') is not None else '-'} "
                f"ordering_needed={int(bool(result.ordering_issues))} "
                f"ordering_repaired={int(bool(result.ordering_repairs))} "
                f"tail_needed={int(bool(result.tail_issues))} "
                f"tail_repaired={int(bool(result.tail_repairs))}",
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
    # Claude Code advertises brotli and zstd in addition to standard-library
    # encodings. Request gzip so the response remains compressed and can also
    # be inspected without adding third-party dependencies.
    headers["Accept-Encoding"] = "gzip"
    headers["Via"] = f"1.1 claude-cache-gateway/{VERSION}"
    return headers


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("observe", "repair"), default="observe", help="observe leaves request bodies byte-identical; repair orders TTLs and relocates a trailing total_tokens marker")
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
    config = Config(upstream=upstream, mode=args.mode, logger=logger, quiet=args.quiet, timeout=args.timeout)
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
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
