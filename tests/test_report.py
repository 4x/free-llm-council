"""Tests for self-contained HTML report rendering and export."""

from __future__ import annotations

import json
import re

import pytest

from llm_council import ModelAnswer, Settings, StorageError, build_report_html, export_html
from llm_council.report import (
    KATEX_FONT_DIR,
    MARKDOWN_RENDERER,
    inline_katex_stylesheet,
)
from llm_council.storage import RunStorage

LATEX_SYNTHESIS = (
    "Inline $f(x) \\propto (x-x_1)^{p-1}(x_2-x)^{q-1}$ renders in the line.\n\n"
    "$$f(v_x)\\propto\\left(1-\\frac{mv_x^2}{2E}\\right)^{(3N-4)/2},\\qquad "
    "|v_x|<\\sqrt{2E/m},$$\n\n"
    "```latex\n$f(x)$ stays literal in code\n```\n"
)


def _run(**overrides: object) -> dict:
    run: dict = {
        "id": "run-1",
        "created_at": 1_792_075_913.0,
        "prompt": "Derive the extended beta density",
        "synthesis": LATEX_SYNTHESIS,
        "judge": "test/beta:free",
        "judge_requested": "most-intelligent",
        "answers": [
            {
                "requested_model": "test/alpha:free",
                "model": "test/alpha:free",
                "content": "Answer with $a^2 + b^2 = c^2$",
                "error": None,
            },
            {
                "requested_model": "test/gamma:free",
                "model": "test/gamma:free",
                "content": None,
                "error": "HTTP 503",
            },
        ],
    }
    run.update(overrides)
    return run


def _markup_only(html: str) -> str:
    r"""Strip inlined script bodies.

    A ``src=`` inside JavaScript source is a string literal, not markup, and the
    browser will never treat it as an attribute - KaTeX ships an unused
    ``\includegraphics`` code path that builds ``<img src=...>`` markup, and that
    path is disabled by ``trust: false``. Stripping the scripts keeps the
    self-containment check pointed at what the document actually references.
    """
    return re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL)


def test_build_report_html_is_fully_self_contained():
    """The report references nothing outside itself, so it renders with no network."""
    html = build_report_html(_run())
    markup = _markup_only(html)

    assert re.findall(r"<(?:script|img|iframe)[^>]*\ssrc=\"([^\"]+)\"", markup) == []
    assert re.findall(r"<link[^>]*\shref=\"([^\"]+)\"", markup) == []
    assert "@import" not in markup
    non_data_urls = [
        target
        for target in re.findall(r"url\(([^)]*)\)", markup)
        if not target.strip("'\"").startswith("data:")
    ]
    assert non_data_urls == []
    # No absolute URL of any kind survives in the markup or stylesheets.
    assert re.findall(r"https?://", markup) == []
    # Every KaTeX font face is inlined as a woff2 data URI.
    assert html.count("url(data:font/woff2;base64,") == 20


def test_build_report_html_inlines_the_shared_renderer_verbatim():
    """The report embeds byte-identical renderer source, so it cannot drift from the UI."""
    html = build_report_html(_run())

    assert MARKDOWN_RENDERER.read_text(encoding="utf-8") in html
    assert "renderToString" in html  # KaTeX engine is inlined too


def test_build_report_html_ships_raw_markdown_for_client_side_typesetting():
    """Rendering happens in the browser via the shared renderer, not pre-rendered server-side."""
    html = build_report_html(_run())

    payload = json.loads(html.split('id="report-data"')[1].split(">", 1)[1].split("</script>")[0])
    assert payload["synthesis"] == LATEX_SYNTHESIS
    assert "propto (x-x_1)^{p-1}" in payload["synthesis"]
    # Nothing is typeset at build time; KaTeX spans only appear once the page runs.
    assert '<span class="katex">' not in html


def test_build_report_html_payload_cannot_close_its_script_element():
    """Hostile synthesis text cannot break out of the embedded JSON payload."""
    hostile = "before </script><img src=x onerror=alert(1)> after $x$"
    html = build_report_html(_run(synthesis=hostile))

    payload = html.split('id="report-data"')[1].split("</script>")[0]
    assert "</script" not in payload
    assert "<img" not in payload
    # The escapes are valid JSON, so the value survives a round trip unchanged.
    decoded = json.loads(payload.split(">", 1)[1])
    assert decoded["synthesis"] == hostile


def test_build_report_html_shows_the_prompt_as_a_rendered_document():
    """The prompt is rendered through the shared document view, not printed as raw text."""
    html = build_report_html(_run(prompt="Derive the extended beta density"))
    markup = _markup_only(html)

    assert '<div class="markdown-body" id="report-prompt"></div>' in markup
    # A pasted LaTeX document is rewritten before it is rendered, exactly as in the UI.
    assert "renderLatexDocument(data.prompt)" in html
    # The old plain-text prompt block is gone: the same text is now typeset.
    assert '<pre class="prompt-text" id="report-prompt">' not in markup


def test_build_report_html_keeps_the_raw_prompt_beside_the_rendered_one():
    """Rewriting a document is auditable: the stored text stays available verbatim.

    The source is written into the document by script, from the JSON payload, so the
    markup never contains it. That keeps the self-containment check pointed at what the
    file references rather than at arbitrary prompt text.
    """
    prompt = "Derive $f(x)$ for </script><b>every</b> case"
    html = build_report_html(_run(prompt=prompt))

    assert '<details class="raw-source" id="report-prompt-raw"></details>' in _markup_only(html)
    assert "rawBody.textContent = data.prompt" in html
    assert "Raw prompt source (" in html
    # The hostile prompt never becomes markup, and the title copy is escaped.
    assert prompt not in _markup_only(html)
    assert "<b>every</b>" not in _markup_only(html)


def test_build_report_html_noscript_fallback_carries_the_raw_synthesis():
    """Without JavaScript the file still shows the synthesis source instead of nothing."""
    html = build_report_html(_run())

    noscript = html.split("<noscript>", 1)[1].split("</noscript>", 1)[0]
    assert "JavaScript is disabled" in noscript
    assert "propto (x-x_1)^{p-1}" in noscript


def test_build_report_html_tolerates_runs_without_synthesis_or_answers():
    """A run missing its synthesis or answers still produces a complete document."""
    html = build_report_html(_run(synthesis=None, answers=[]))

    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<noscript>") == html.count("</noscript>") == 1
    # A run that simply never stored member answers must not read like a failure.
    assert "no member answers stored" in html
    assert "0 of 0 answers usable" not in html


def test_build_report_html_truncates_long_titles_on_a_word_boundary():
    """Long prompts produce a readable title instead of one cut mid-word."""
    html = build_report_html(_run(prompt="word " * 40))

    title = html.split("<h1>", 1)[1].split("</h1>", 1)[0]
    assert title.endswith("…")
    assert "  " not in title
    assert not title.removesuffix("…").endswith("wor")


def test_inline_katex_stylesheet_rewrites_every_font_face():
    """The vendored stylesheet's font faces all become data URIs."""
    css = inline_katex_stylesheet()

    assert css.count("url(data:font/woff2;base64,") == 20
    assert "url(fonts/" not in css
    assert 'format("truetype")' not in css


def test_inline_katex_stylesheet_fails_loudly_when_only_woff2_is_missing(monkeypatch):
    """A missing vendored font is reported instead of silently emitting a broken report."""
    missing = KATEX_FONT_DIR / "KaTeX_Main-Regular.woff2"

    def fail_read(_self: object) -> bytes:
        raise OSError(f"fixture unreadable: {missing.name}")

    monkeypatch.setattr(type(missing), "read_bytes", fail_read)
    with pytest.raises(StorageError, match="Cannot read bundled font"):
        inline_katex_stylesheet()


def _stored_settings(tmp_path):
    return Settings().with_overrides(
        save_prompt="duckdb",
        save_initial_answers="duckdb",
        save_synthesis="duckdb",
        duckdb_path=tmp_path / "data.duckdb",
        markdown_dir=tmp_path / "markdown",
        html_dir=tmp_path / "html",
    )


def _persist_run(settings, question_id: str, prompt: str, synthesis: str) -> None:
    answer = ModelAnswer(
        "openrouter/free",
        "test/alpha:free",
        content="Member answer",
        usage={
            "generation_id": "gen-member-001",
            "provider": "Fireworks",
            "strategy": "fallback",
            "attempt": 2,
            "generation_time": 1250,
            "elapsed_ms": 1300,
            "prompt_tokens": 40,
            "completion_tokens": 160,
            "total_tokens": 200,
            "completion_tokens_details": {"reasoning_tokens": 60},
            "prompt_tokens_details": {"cached_tokens": 20},
            "cost": 0.004,
            "cost_details": {"server_tool_cost": 0.004},
            "server_tool_use": {"web_search_requests": 1},
            "citations": [
                {
                    "url": "https://example.com/beta-dist",
                    "title": "Beta Distribution Reference",
                    "content": "Conjugate prior details",
                }
            ],
            "generation": {
                "native_tokens_prompt": 38,
                "native_tokens_completion": 155,
                "upstream_inference_cost": 0.0021,
                "cache_discount": 0.0004,
            },
            "generation_reconciled_at": 1790000000,
        },
    )
    synth_answer = ModelAnswer(
        "test/alpha:free",
        "test/alpha:free",
        content=synthesis,
        usage={
            "generation_id": "gen-synth-002",
            "provider": "Chutes",
            "elapsed_ms": 2100,
            "total_tokens": 512,
        },
    )
    with RunStorage(settings, question_id, prompt) as storage:
        storage.save_question(prompt, None)
        storage.save_answer(answer, "initial", 1)
        storage.save_answer(synth_answer, "synthesis")


def test_export_html_writes_one_self_contained_report_per_stored_run(tmp_path):
    """export_html writes a standalone .html file for each stored synthesis."""
    settings = _stored_settings(tmp_path)
    _persist_run(settings, "q1", "Explain the beta distribution", LATEX_SYNTHESIS)

    exported = export_html(settings=settings)

    assert len(exported) == 1
    report = exported[0]
    assert report.parent == settings.html_dir
    assert report.suffix == ".html"
    assert "explain-beta-distribution" in report.name
    content = report.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "propto (x-x_1)^{p-1}" in content
    assert "gen-member-001" in content
    assert "gen-synth-002" in content
    assert "Fireworks" in content
    assert "https://example.com/beta-dist" in content
    # Reconciled GET /generation accounting is embedded and rendered offline.
    assert '"upstream_inference_cost": 0.0021' in content
    assert '"native_tokens_prompt": 38' in content
    assert "Native prompt: " in content
    assert "upstream provider cost" in content.lower()
    assert "buildUsageElement" in content
    assert "buildCitationsElement" in content
    assert _markup_only(content).count("url(data:font/woff2;base64,") == 20


def test_export_html_is_idempotent_and_skips_runs_without_synthesis(tmp_path):
    """Re-exporting writes nothing new, and synthesis-less runs are never exported."""
    settings = _stored_settings(tmp_path)
    _persist_run(settings, "q1", "Explain the beta distribution", LATEX_SYNTHESIS)
    prompt_only = ModelAnswer("test/alpha:free", "test/alpha:free", content="Unused")
    with RunStorage(settings, "q2", "Prompt without a synthesis") as storage:
        storage.save_question("Prompt without a synthesis", None)
        storage.save_answer(prompt_only, "initial", 1)

    first = export_html(settings=settings)
    assert len(first) == 1

    assert export_html(settings=settings) == ()
    assert [path.name for path in settings.html_dir.iterdir()] == [first[0].name]


def test_export_html_can_target_a_single_run(tmp_path):
    """Passing run_id restricts the export to that stored run."""
    settings = _stored_settings(tmp_path)
    _persist_run(settings, "q1", "First stored question", LATEX_SYNTHESIS)
    _persist_run(settings, "q2", "Second stored question", LATEX_SYNTHESIS)

    exported = export_html(settings=settings, run_id="q2")

    assert len(exported) == 1
    assert "second-stored-question" in exported[0].name
    assert [path.name for path in settings.html_dir.iterdir()] == [exported[0].name]


def test_export_html_requires_an_existing_duckdb_file(tmp_path):
    """A missing DuckDB file is reported rather than producing an empty export."""
    settings = _stored_settings(tmp_path)

    with pytest.raises(StorageError, match="does not exist"):
        export_html(settings=settings)


def test_export_html_honours_html_dir_override(tmp_path):
    """The html_dir setting controls where reports land."""
    settings = _stored_settings(tmp_path)
    _persist_run(settings, "q1", "Explain the beta distribution", LATEX_SYNTHESIS)
    elsewhere = tmp_path / "reports"

    exported = export_html(settings=settings, html_dir=elsewhere)

    assert len(exported) == 1
    assert exported[0].parent == elsewhere
    assert not settings.html_dir.exists()
