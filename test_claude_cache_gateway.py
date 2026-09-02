#!/usr/bin/env python3
import http.client
import json
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import claude_cache_gateway as gateway


class RepairTests(unittest.TestCase):
    def fixture(self):
        return {
            "model": "claude-test",
            "system": [
                {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ],
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "read", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t", "content": "private result"},
                        {
                            "type": "text",
                            "text": gateway.CONTINUATION_PROMPTS.copy().pop(),
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": gateway.NO_RESPONSE}]},
                {"role": "user", "content": gateway.CONTINUATION_PROMPTS.copy().pop()},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "<total_tokens>15000000 tokens left</total_tokens>",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                },
            ],
        }

    def test_observe_is_byte_identical(self):
        raw = json.dumps(self.fixture(), indent=2).encode()
        result = gateway.analyze_and_maybe_repair(raw, repair=False)
        self.assertEqual(result.body, raw)
        self.assertGreaterEqual(len(result.detected), 2)
        self.assertEqual(result.repairs, [])

    def test_repair_moves_markers_to_stable_tool_result(self):
        raw = gateway.compact_json(self.fixture())
        before = gateway.ordered_breakpoints(json.loads(raw))
        result = gateway.analyze_and_maybe_repair(raw, repair=True)
        repaired = json.loads(result.body)
        after = gateway.ordered_breakpoints(repaired)
        self.assertTrue(any(x["kind"] == "total_tokens_reminder" for x in before))
        self.assertFalse(any(x["kind"] in {"total_tokens_reminder", "continuation_prompt"} for x in after))
        tool_result = repaired["messages"][1]["content"][0]
        self.assertEqual(tool_result["cache_control"]["ttl"], "5m")
        self.assertNotIn("cache_control", repaired["messages"][-1]["content"][0])
        self.assertEqual(len(result.repairs), 2)

    def test_historical_continuation_text_is_not_rewritten(self):
        body = self.fixture()
        historical = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": next(iter(gateway.CONTINUATION_PROMPTS)),
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
        }
        body["messages"].insert(0, historical)
        result = gateway.analyze_and_maybe_repair(gateway.compact_json(body), repair=True)
        repaired = json.loads(result.body)
        self.assertIn("cache_control", repaired["messages"][0]["content"][0])

    def test_plain_request_without_bug_is_unchanged(self):
        raw = b'{"model":"m","messages":[{"role":"user","content":"hello"}]}'
        result = gateway.analyze_and_maybe_repair(raw, repair=True)
        self.assertEqual(result.body, raw)
        self.assertEqual(result.detected, [])


class MemoryLogger:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


class IntegrationTests(unittest.TestCase):
    def test_streaming_proxy_forwards_auth_repairs_and_logs_usage(self):
        observed = {}

        class UpstreamHandler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                pass

            def do_POST(self):  # noqa: N802
                length = int(self.headers["Content-Length"])
                observed["authorization"] = self.headers.get("Authorization")
                observed["body"] = json.loads(self.rfile.read(length))
                body = (
                    b'event: message_start\n'
                    b'data: {"type":"message_start","message":{"usage":{"input_tokens":2,"output_tokens":1,"cache_read_input_tokens":10,"cache_creation_input_tokens":20}}}\n\n'
                    b'event: message_delta\n'
                    b'data: {"type":"message_delta","usage":{"output_tokens":7,"output_tokens_details":{"thinking_tokens":3}}}\n\n'
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        logger = MemoryLogger()
        config = gateway.Config(
            upstream=urllib.parse.urlsplit(f"http://127.0.0.1:{upstream.server_port}"),
            mode="repair",
            logger=logger,
            quiet=True,
            timeout=5,
        )
        proxy = gateway.CacheGatewayServer(("127.0.0.1", 0), config)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        try:
            fixture = RepairTests().fixture()
            connection = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=5)
            connection.request(
                "POST",
                "/v1/messages",
                body=gateway.compact_json(fixture),
                headers={"Content-Type": "application/json", "Authorization": "Bearer private"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
        finally:
            proxy.shutdown()
            upstream.shutdown()
            proxy.server_close()
            upstream.server_close()
        self.assertEqual(observed["authorization"], "Bearer private")
        self.assertFalse(
            any(
                point["kind"] in {"total_tokens_reminder", "continuation_prompt"}
                for point in gateway.ordered_breakpoints(observed["body"])
            )
        )
        self.assertEqual(len(logger.records), 1)
        self.assertEqual(logger.records[0]["usage"]["output_tokens"], 7)
        self.assertEqual(logger.records[0]["usage"]["reasoning_tokens"], 3)
        serialized_log = json.dumps(logger.records[0])
        self.assertNotIn("private", serialized_log)
        self.assertNotIn("private result", serialized_log)


class UsageTests(unittest.TestCase):
    def test_fragmented_sse_usage(self):
        body = (
            b'event: message_start\n'
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":2,"cache_read_input_tokens":10,"cache_creation_input_tokens":20,"output_tokens":1}}}\n\n'
            b'event: message_delta\n'
            b'data: {"type":"message_delta","usage":{"output_tokens":7,"output_tokens_details":{"thinking_tokens":3},"cache_creation":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":16}}}\n\n'
        )
        observer = gateway.UsageObserver()
        for i in range(0, len(body), 7):
            observer.feed(body[i : i + 7])
        usage = observer.finish()
        self.assertEqual(
            usage,
            {
                "input_tokens": 2,
                "output_tokens": 7,
                "cache_read_tokens": 10,
                "cache_write_tokens": 20,
                "cache_write_5m_tokens": 4,
                "cache_write_1h_tokens": 16,
                "reasoning_tokens": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
