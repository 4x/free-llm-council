from __future__ import annotations

import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest


def model(model_id: str, price: str = "0", context: int = 32000) -> dict[str, Any]:
    return {
        "id": model_id,
        "pricing": {"prompt": price, "completion": price, "output": price, "request": "0"},
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "context_length": context,
    }


class MockRouter:
    def __init__(self) -> None:
        self.catalog = [
            model("test/alpha:free"),
            model("test/beta:free", context=64000),
            model("test/paid", "0.01"),
            model("openrouter/free"),
        ]
        self.ranked = [self.catalog[2], self.catalog[1], self.catalog[0]]
        self.calls: list[dict[str, Any]] = []
        self.gets: list[dict[str, list[str]]] = []
        self.fail_models: set[str] = set()
        self.judge_failure = False
        self.page_size = 1000
        self.reply: dict[str, Any] | None = None
        self.model_replies: dict[str, dict[str, Any]] = {}
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def send(self, payload: dict[str, Any], status: int = 200) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                params = parse_qs(urlsplit(self.path).query)
                state.gets.append(params)
                catalog = state.ranked if "sort" in params else state.catalog
                # Honor the catalog price caps so free-only queries really hide paid models.
                if "max_price" in params:
                    cap = Decimal(params["max_price"][0])
                    catalog = [
                        entry
                        for entry in catalog
                        if Decimal(str(entry["pricing"]["prompt"])) <= cap
                    ]
                if "max_output_price" in params:
                    cap = Decimal(params["max_output_price"][0])
                    catalog = [
                        entry
                        for entry in catalog
                        if Decimal(str(entry["pricing"]["completion"])) <= cap
                    ]
                offset = int(params.get("offset", ["0"])[0])
                page = catalog[offset : offset + state.page_size]
                more = offset + len(page) < len(catalog)
                self.send(
                    {
                        "data": page,
                        "total_count": len(catalog),
                        "links": {"next": "/models?next" if more else None},
                    }
                )

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                body["authorization"] = self.headers.get("Authorization")
                body["openrouter_metadata_header"] = self.headers.get("X-OpenRouter-Metadata")
                state.calls.append(body)
                is_judge = body["messages"][0]["role"] == "system"
                if body["model"] in state.fail_models or (is_judge and state.judge_failure):
                    self.send({"error": {"code": 503, "message": "fixture unavailable"}}, 503)
                    return
                if not is_judge and body["model"] in state.model_replies:
                    self.send(state.model_replies[body["model"]])
                    return
                if state.reply is not None:
                    self.send(state.reply)
                    return
                text = "Synthesized fixture answer" if is_judge else "Answer: " + body["model"]
                actual = (
                    "test/alpha:free" if body["model"].startswith("openrouter/") else body["model"]
                )
                self.send(
                    {
                        "id": "gen-fixture-synth" if is_judge else f"gen-fixture-{body['model']}",
                        "model": actual,
                        "choices": [
                            {
                                "message": {"content": text},
                                "finish_reason": "stop",
                                "native_finish_reason": "STOP",
                            }
                        ],
                        "usage": {"total_tokens": 42},
                        "openrouter_metadata": {
                            "strategy": "direct",
                            "region": "us-east",
                            "attempt": 1,
                            "generation_time": 120,
                            "endpoints": [{"provider": "FixtureProvider", "selected": True}],
                        },
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)


@pytest.fixture
def router():
    state = MockRouter()
    state.thread.start()
    try:
        yield state
    finally:
        state.server.shutdown()
        state.server.server_close()
        state.thread.join()


@pytest.fixture
def config(router, tmp_path):
    from llm_council import Settings

    return Settings().with_overrides(
        api_key="fixture-key",
        base_url=router.url,
        request_interval=0,
        max_retries=0,
        duckdb_path=tmp_path / "council.duckdb",
        markdown_dir=tmp_path / "markdown",
        judge="test/beta:free",
        free_router_model="openrouter/free",
        auto_router_model="openrouter/auto",
        router_models=("openrouter/free",),
    )
