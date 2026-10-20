"""Tests for the interactive HTML web interface and HTTP/SSE API."""

from __future__ import annotations

import json
import re
import threading
from threading import Event
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from llm_council import CouncilError, create_web_server, load_history, query_result


@pytest.fixture()
def web_url(config, router):
    """Run create_web_server on an ephemeral localhost port."""
    server = create_web_server(
        host="127.0.0.1",
        port=0,
        settings=config.with_overrides(
            save_prompt="duckdb",
            save_initial_answers="duckdb",
            save_synthesis="duckdb",
        ),
    )
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _parse_sse(raw_body: str) -> list[dict]:
    events: list[dict] = []
    for block in raw_body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


def test_web_index_serves_required_ui_elements(web_url):
    """GET / returns the complete HTML workspace with all requested interactive controls."""
    with urlopen(f"{web_url}/", timeout=5) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")

    assert 'id="prompt-input"' in html
    assert 'id="clear-btn"' in html
    assert 'id="stop-btn"' in html
    assert "btn-danger" in html
    assert 'id="new-btn"' in html
    assert 'id="followup-btn"' in html
    assert "data-tooltip=" in html
    assert 'id="attach-file-btn"' in html
    assert "📎" in html
    assert 'id="judge-select"' in html
    assert 'id="search-web-checkbox"' in html
    assert 'id="search-web-toggle"' in html
    assert 'id="pagination-bar"' in html
    assert 'id="session-history-list"' in html
    assert 'id="db-history-list"' in html
    assert "function renderUsagePillsHtml(usage)" in html
    assert "function renderCitationsHtml(usage)" in html
    assert "citations-strip" in html
    assert "meta-pill" in html


def test_web_config_and_history_endpoints(config, router, web_url):
    """GET /api/config and GET /api/history return judge strategies and stored DuckDB runs."""
    query_result(
        "Persisted DB question",
        settings=config,
        save_prompt="duckdb",
        save_initial_answers="duckdb",
        save_synthesis="duckdb",
    )

    with urlopen(f"{web_url}/api/config", timeout=5) as cfg_resp:
        cfg = json.loads(cfg_resp.read().decode("utf-8"))
    assert cfg["judge"] == "test/beta:free"
    assert "most-intelligent" in cfg["strategies"]
    assert "largest-context" in cfg["strategies"]

    with urlopen(f"{web_url}/api/history", timeout=5) as hist_resp:
        hist = json.loads(hist_resp.read().decode("utf-8"))
    assert len(hist["history"]) == 1
    entry = hist["history"][0]
    assert entry["prompt"] == "Persisted DB question"
    assert entry["synthesis"] == "Synthesized fixture answer"
    assert entry["synthesis_usage"]["generation_id"] == "gen-fixture-synth"
    assert entry["synthesis_usage"]["provider"] == "FixtureProvider"
    assert len(entry["answers"]) == 2
    assert entry["answers"][0]["usage"]["provider"] == "FixtureProvider"
    assert entry["source"] == "db"


def test_web_query_streams_member_answers_then_synthesis(router, web_url):
    """POST /api/query streams discovered, member_answer, synthesizing, and synthesis."""
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps(
            {
                "prompt": "Explain quantum tunneling",
                "file_name": "notes.txt",
                "file_content": "Include WKB approximation",
                "judge": "largest-context",
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        events = _parse_sse(resp.read().decode("utf-8"))

    event_types = [evt["type"] for evt in events]
    assert event_types[0] == "discovered"
    assert event_types.count("member_answer") == 2
    assert "synthesizing" in event_types
    assert event_types[-1] == "synthesis"

    synth_evt = events[-1]
    assert synth_evt["synthesis"] == "Synthesized fixture answer"
    assert synth_evt["judge"] == "test/beta:free"
    assert synth_evt["usage"]["generation_id"] == "gen-fixture-synth"
    assert synth_evt["usage"]["provider"] == "FixtureProvider"
    assert all(call["openrouter_metadata_header"] == "enabled" for call in router.calls)
    assert (
        router.calls[0]["messages"][0]["content"]
        == "Explain quantum tunneling\n\nInclude WKB approximation"
    )


def test_web_config_reports_search_web_default(web_url):
    """GET /api/config exposes the server's web-search default for the checkbox."""
    with urlopen(f"{web_url}/api/config", timeout=5) as cfg_resp:
        cfg = json.loads(cfg_resp.read().decode("utf-8"))
    assert cfg["search_web"] is False


def test_web_query_search_web_adds_server_tools_to_every_call(router, web_url):
    """POST /api/query with search_web sends OpenRouter's web search tool."""
    for entry in router.catalog:
        entry["supported_parameters"] = ["tools"]
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps(
            {"prompt": "What happened today?", "judge": "test/beta:free", "search_web": True}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:
        events = _parse_sse(resp.read().decode("utf-8"))

    assert events[-1]["type"] == "synthesis"
    assert len(router.calls) == 3
    assert all(
        call["tools"] == [{"type": "openrouter:web_search", "parameters": {"max_uses": 3}}]
        for call in router.calls
    )


def test_web_query_rejects_non_boolean_search_web(web_url):
    """POST /api/query validates the search_web checkbox payload."""
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps({"prompt": "Question", "search_web": "yes"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req, timeout=5)
    assert exc_info.value.code == 400


def test_web_query_followup_attaches_selected_conversation_context(router, web_url):
    """POST /api/query with prior_turns sends the selected conversation's prompt and synthesis."""
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps(
            {
                "prompt": "How does that apply to alpha decay?",
                "judge": "test/beta:free",
                "prior_turns": [
                    ["Selected earlier prompt", "Selected earlier synthesis"],
                ],
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:
        events = _parse_sse(resp.read().decode("utf-8"))

    assert events[-1]["type"] == "synthesis"
    assert router.calls[0]["messages"] == [
        {"role": "user", "content": "Selected earlier prompt"},
        {"role": "assistant", "content": "Selected earlier synthesis"},
        {"role": "user", "content": "How does that apply to alpha decay?"},
    ]


def test_web_improve_endpoint_returns_improved_prompts(router, web_url):
    """POST /api/improve returns numbered prompt suggestions from a single LLM."""
    router.reply = {
        "model": "test/beta:free",
        "choices": [
            {
                "message": {"content": '["Sharper prompt 1", "Sharper prompt 2"]'},
                "finish_reason": "stop",
            }
        ],
    }
    req = Request(
        f"{web_url}/api/improve",
        data=json.dumps({"prompt": "Rough question", "judge": "test/beta:free"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    assert payload["prompts"] == ["Sharper prompt 1", "Sharper prompt 2"]
    assert len(router.calls) == 1


def test_web_empty_query_returns_400(web_url):
    """POST /api/query with an empty prompt and no file returns HTTP 400."""
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps({"prompt": "   "}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req, timeout=5)
    assert exc_info.value.code == 400


def test_web_index_references_vendored_katex(web_url):
    """GET / wires up the locally vendored KaTeX bundle so LaTeX renders offline."""
    with urlopen(f"{web_url}/", timeout=5) as resp:
        html = resp.read().decode("utf-8")

    assert 'href="/static/katex/katex.min.css"' in html
    assert "math-display" in html

    # Both script tags must be blocking head loads: the inline app script renders
    # stored history immediately, so window.katex and the shared renderer have to
    # already be defined by the time it runs. defer/async would break that.
    script_tags = re.findall(r"<script[^>]*\ssrc=\"[^\"]+\"[^>]*>", html)
    sources = " ".join(script_tags)
    assert "katex.min.js" in sources
    assert "markdown_renderer.js" in sources
    assert all("defer" not in tag and "async" not in tag for tag in script_tags), script_tags


def test_web_serves_shared_markdown_renderer(web_url):
    """GET /static/markdown_renderer.js serves the renderer shared with HTML reports."""
    with urlopen(f"{web_url}/static/markdown_renderer.js", timeout=5) as resp:
        assert resp.status == 200
        assert "text/javascript" in resp.headers.get("Content-Type", "")
        source = resp.read().decode("utf-8")

    for name in (
        "escapeHtml",
        "renderMathToHtml",
        "renderInlineMarkdown",
        "renderMarkdown",
        "sanitizeLatexDocument",
        "renderLatexDocument",
    ):
        assert f"function {name}" in source, name
    assert "window.katex" in source


def test_web_index_pages_a_rendered_prompt_alongside_synthesis_and_answers(web_url):
    """A stored prompt gets its own page, rendered as a document with its source beside it."""
    with urlopen(f"{web_url}/", timeout=5) as resp:
        html = resp.read().decode("utf-8")

    assert 'key: "prompt"' in html
    assert "renderPromptCard(run)" in html
    assert "function renderPromptCard(run)" in html
    # The page key has to survive the string/number switch the tabs share with answers.
    assert 'rawKey === "prompt"' in html
    # Rendered output comes from the shared document view, and the raw text can be seen.
    assert "renderLatexDocument(raw)" in html
    assert "Raw prompt source" in html


def test_web_serves_vendored_katex_assets(web_url):
    """GET /static serves the vendored KaTeX stylesheet, script, and woff2 fonts."""
    for asset_path, expected_type in (
        ("/static/katex/katex.min.css", "text/css"),
        ("/static/katex/katex.min.js", "text/javascript"),
        ("/static/katex/fonts/KaTeX_Main-Regular.woff2", "font/woff2"),
    ):
        with urlopen(f"{web_url}{asset_path}", timeout=5) as resp:
            assert resp.status == 200, asset_path
            assert expected_type in resp.headers.get("Content-Type", ""), asset_path
            assert len(resp.read()) > 0, asset_path


def test_web_static_rejects_paths_outside_static_dir(web_url):
    """GET /static refuses traversal attempts instead of serving package sources."""
    for hostile_path in ("/static/../web.py", "/static/%2e%2e%2fweb.py"):
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{web_url}{hostile_path}", timeout=5)
        assert exc_info.value.code == 404, hostile_path


def test_load_history_returns_empty_when_db_missing(config):
    """load_history returns an empty list when the DuckDB file does not exist yet."""
    assert load_history(settings=config) == []


def test_query_result_respects_cancel_event(config, router):
    """Setting cancel_event aborts in-progress council retrieval before synthesis."""
    cancel = Event()

    def stop_on_first_answer(evt: dict) -> None:
        if evt.get("type") == "member_answer":
            cancel.set()

    with pytest.raises(CouncilError, match="aborted"):
        query_result(
            "Abort this council run",
            settings=config.with_overrides(max_concurrency=1),
            on_event=stop_on_first_answer,
            cancel_event=cancel,
        )
