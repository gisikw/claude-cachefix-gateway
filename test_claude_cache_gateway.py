#!/usr/bin/env python3
import gzip
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
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "private prompt",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        }
                    ],
                },
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

    def test_observe_is_byte_identical_and_reports_both_issues(self):
        raw = json.dumps(self.fixture(), indent=2).encode()
        result = gateway.analyze_and_maybe_repair(raw, repair=False)
        self.assertEqual(result.body, raw)
        self.assertEqual(len(result.ordering_issues), 1)
        self.assertEqual(len(result.tail_issues), 1)
        self.assertEqual(result.ordering_repairs, [])
        self.assertEqual(result.tail_repairs, [])

    def test_repair_orders_ttls_and_removes_tail_marker(self):
        raw = gateway.compact_json(self.fixture())
        result = gateway.analyze_and_maybe_repair(raw, repair=True)
        repaired = json.loads(result.body)
        self.assertEqual(len(result.ordering_repairs), 1)
        self.assertEqual(len(result.tail_repairs), 1)
        self.assertEqual(repaired["messages"][0]["content"][0]["cache_control"]["ttl"], "5m")
        self.assertNotIn("cache_control", repaired["messages"][-1]["content"][0])
        self.assertEqual(gateway.ttl_order_issues(repaired), [])

    def test_tail_marker_moves_to_nearest_ordinary_turn(self):
        body = self.fixture()
        del body["messages"][0]["content"][0]["cache_control"]
        result = gateway.analyze_and_maybe_repair(gateway.compact_json(body), repair=True)
        repaired = json.loads(result.body)
        target = repaired["messages"][0]["content"][0]
        self.assertEqual(target["cache_control"]["ttl"], "1h")
        self.assertEqual(result.ordering_repairs, [])
        self.assertEqual(len(result.tail_repairs), 1)

    def test_continuation_content_and_marker_are_not_special_cased(self):
        body = self.fixture()
        continuation = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<tool-result>Tool call complete. Results are above.</tool-result>",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
        }
        body["messages"].insert(1, continuation)
        result = gateway.analyze_and_maybe_repair(gateway.compact_json(body), repair=True)
        repaired = json.loads(result.body)
        block = repaired["messages"][1]["content"][0]
        self.assertEqual(block["text"], continuation["content"][0]["text"])
        self.assertEqual(block["cache_control"]["ttl"], "5m")

    def test_plain_request_without_bug_is_unchanged(self):
        raw = b'{"model":"m","messages":[{"role":"user","content":"hello"}]}'
        result = gateway.analyze_and_maybe_repair(raw, repair=True)
        self.assertEqual(result.body, raw)
        self.assertEqual(result.ordering_issues, [])
        self.assertEqual(result.tail_issues, [])


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
                observed["accept_encoding"] = self.headers.get("Accept-Encoding")
                observed["body"] = json.loads(self.rfile.read(length))
                body = gzip.compress(
                    (
                    b'event: message_start\n'
                    b'data: {"type":"message_start","message":{"usage":{"input_tokens":2,"output_tokens":1,"cache_read_input_tokens":10,"cache_creation_input_tokens":20}}}\n\n'
                    b'event: message_delta\n'
                    b'data: {"type":"message_delta","usage":{"output_tokens":7,"output_tokens_details":{"thinking_tokens":3}}}\n\n'
                    )
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        logger = MemoryLogger()
        config = gateway.Config(
            upstream=urllib.parse.urlsplit(f"http://127.0.0.1:{upstream.server_port}"),
            mode="repair",
            logger=logger,
            quiet=True,
            timeout=5,
        )
        proxy = gateway.CacheGatewayServer(("127.0.0.1", 0), config)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=5)
            connection.request(
                "POST",
                "/v1/messages",
                body=gateway.compact_json(RepairTests().fixture()),
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
        self.assertEqual(observed["accept_encoding"], "gzip")
        self.assertEqual(gateway.ttl_order_issues(observed["body"]), [])
        self.assertFalse(any(point["kind"] == "total_tokens_reminder" for point in gateway.ordered_breakpoints(observed["body"])))
        self.assertEqual(len(logger.records), 1)
        record = logger.records[0]
        self.assertEqual(record["usage"]["output_tokens"], 7)
        self.assertEqual(record["usage"]["reasoning_tokens"], 3)
        self.assertEqual(len(record["ordering_repairs"]), 1)
        self.assertEqual(len(record["tail_repairs"]), 1)
        serialized_log = json.dumps(record)
        self.assertNotIn("private", serialized_log)
        self.assertNotIn("private prompt", serialized_log)


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
        expected = {
            "input_tokens": 2,
            "output_tokens": 7,
            "cache_read_tokens": 10,
            "cache_write_tokens": 20,
            "cache_write_5m_tokens": 4,
            "cache_write_1h_tokens": 16,
            "reasoning_tokens": 3,
        }
        self.assertEqual(observer.finish(), expected)

        compressed = gzip.compress(body)
        observer = gateway.UsageObserver("gzip")
        for i in range(0, len(compressed), 7):
            observer.feed(compressed[i : i + 7])
        self.assertEqual(observer.finish(), expected)


if __name__ == "__main__":
    unittest.main()
