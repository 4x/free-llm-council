"""Self-contained HTML report rendering for stored council runs.

A report is a single ``.html`` file that references nothing outside itself: the
KaTeX stylesheet is inlined with its woff2 fonts converted to data URIs, and both
the KaTeX engine and the shared Markdown renderer are inlined as script text. That
makes the artifact readable, archivable, and printable with no network access and
no copy of llm-council on the machine opening it.

The Markdown renderer is the *same* source the web UI serves
(``static/markdown_renderer.js``); see that file's header for why there is only one
implementation.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import StorageError

logger = logging.getLogger("llm_council")

STATIC_DIR = Path(__file__).resolve().parent / "static"
KATEX_DIR = STATIC_DIR / "katex"
KATEX_STYLESHEET = KATEX_DIR / "katex.min.css"
KATEX_SCRIPT = KATEX_DIR / "katex.min.js"
KATEX_FONT_DIR = KATEX_DIR / "fonts"
MARKDOWN_RENDERER = STATIC_DIR / "markdown_renderer.js"

WOFF2_MIME_TYPE = "font/woff2"
MAX_TITLE_CHARS = 72

# KaTeX ships 20 font families, each declared as a single @font-face whose src list
# offers woff2, woff, and ttf. Only woff2 is vendored and every browser we support
# prefers it, so each src list is rewritten to just the inlined woff2 entry. If the
# upstream stylesheet ever changes shape, the count check below fails loudly rather
# than silently emitting a report that points at font files it cannot ship.
_FONT_FACE_COUNT = 20
_FONT_FACE_RE = re.compile(
    r'src:url\(fonts/(?P<name>[A-Za-z0-9_.-]+)\.woff2\)\s*format\("woff2"\),'
    r'url\([^)]*\)\s*format\("woff"\),'
    r'url\([^)]*\)\s*format\("truetype"\)'
)

# JSON embedded in a <script> element must not be able to close that element, so the
# three characters that could start an HTML construct are written as \u escapes.
_JSON_ESCAPES = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"}

_REPORT_CSS = """
  :root {
    color-scheme: light;
  }

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    padding: 40px 24px 72px;
    background: #ffffff;
    color: #1b1f24;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    font-size: 16px;
    line-height: 1.65;
  }

  .report {
    max-width: 860px;
    margin: 0 auto;
  }

  .report-header {
    border-bottom: 2px solid #d8dee4;
    margin-bottom: 28px;
    padding-bottom: 16px;
  }

  .report-header h1 {
    font-size: 26px;
    margin: 0 0 8px;
    line-height: 1.25;
    overflow-wrap: break-word;
  }

  .report-meta {
    color: #59636e;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12.5px;
    margin: 0;
  }

  .report-meta span + span::before {
    content: " · ";
    color: #b0b8c0;
  }

  h2 {
    font-size: 19px;
    margin: 32px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e4e8ed;
  }

  h3 {
    font-size: 15.5px;
    margin: 24px 0 8px;
  }

  .prompt-text {
    background: #f6f8fa;
    border: 1px solid #d8dee4;
    border-left: 4px solid #6b7280;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
    margin: 0;
    overflow-wrap: break-word;
    padding: 14px 16px;
    white-space: pre-wrap;
  }

  /* The stored prompt text, kept verbatim beside its rendered view. */
  .raw-source {
    margin: 18px 0 0;
  }

  .raw-source summary {
    color: #59636e;
    cursor: pointer;
    font-size: 13px;
  }

  .raw-source pre {
    margin-bottom: 0;
  }

  .answer {
    border: 1px solid #e4e8ed;
    border-radius: 8px;
    margin: 0 0 20px;
    padding: 4px 18px 14px;
  }

  .answer-error {
    color: #a4262c;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
  }

  .markdown-body {
    overflow-wrap: break-word;
  }

  .markdown-body h1,
  .markdown-body h2,
  .markdown-body h3,
  .markdown-body h4 {
    border: none;
    font-size: 17px;
    margin: 20px 0 8px;
    padding: 0;
  }

  .markdown-body h1:first-child,
  .markdown-body h2:first-child,
  .markdown-body h3:first-child,
  .markdown-body p:first-child {
    margin-top: 0;
  }

  .markdown-body code {
    background: #f0f3f6;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.87em;
    padding: 2px 5px;
  }

  .markdown-body pre {
    background: #f6f8fa;
    border: 1px solid #d8dee4;
    border-radius: 6px;
    overflow-x: auto;
    padding: 12px 14px;
  }

  .markdown-body pre code {
    background: none;
    padding: 0;
  }

  .markdown-body blockquote {
    border-left: 3px solid #b0b8c0;
    color: #59636e;
    margin: 12px 0;
    padding: 2px 16px;
  }

  .markdown-body table {
    border-collapse: collapse;
    font-size: 0.94em;
    margin: 14px 0;
    width: 100%;
  }

  .markdown-body th,
  .markdown-body td {
    border: 1px solid #d8dee4;
    padding: 7px 10px;
    text-align: left;
  }

  .markdown-body th {
    background: #f6f8fa;
  }

  .math-display {
    overflow-x: auto;
    padding: 2px 0;
  }

  .markdown-body .katex {
    font-size: 1.06em;
  }

  .report-footer {
    border-top: 1px solid #e4e8ed;
    color: #59636e;
    font-size: 12px;
    margin-top: 40px;
    padding-top: 12px;
  }

  @media print {
    body {
      font-size: 11.5pt;
      padding: 0;
    }

    .report {
      max-width: none;
    }

    .answer,
    .markdown-body pre,
    .markdown-body table,
    .math-display {
      break-inside: avoid;
    }

    .prompt-text {
      background: none;
      border-left: 3px solid #6b7280;
      border-right: none;
      border-top: none;
      border-bottom: none;
    }
  }
"""

_BOOT_SCRIPT = """
(function () {
  var dataElement = document.getElementById("report-data");
  if (!dataElement) return;
  var data;
  try {
    data = JSON.parse(dataElement.textContent || "{}");
  } catch (err) {
    return;
  }

  // renderLatexDocument escapes untrusted content itself, and rewrites a pasted LaTeX
  // document (comments, sectioning, doubled backslashes, bare environments) first.
  function renderInto(element, text) {
    if (element) element.innerHTML = renderLatexDocument(text || "");
  }

  // The prompt is shown as a rendered document view, with the stored text kept
  // verbatim beside it: rewriting a document is a transformation the reader must be
  // able to audit, so the source is never more than one click away.
  var promptHost = document.getElementById("report-prompt");
  if (promptHost) {
    if (data.prompt) {
      promptHost.innerHTML = renderLatexDocument(data.prompt);
    } else {
      var emptyPrompt = document.createElement("p");
      emptyPrompt.className = "report-meta";
      emptyPrompt.textContent = "This run stored no prompt.";
      promptHost.appendChild(emptyPrompt);
    }
  }

  var rawPromptHost = data.prompt ? document.getElementById("report-prompt-raw") : null;
  if (rawPromptHost) {
    var rawSummary = document.createElement("summary");
    rawSummary.textContent = "Raw prompt source (" + data.prompt.length + " characters)";
    var rawBody = document.createElement("pre");
    rawBody.className = "prompt-text";
    rawBody.textContent = data.prompt;
    rawPromptHost.appendChild(rawSummary);
    rawPromptHost.appendChild(rawBody);
  }

  renderInto(document.getElementById("report-synthesis"), data.synthesis);

  var answers = data.answers || [];
  var countElement = document.getElementById("report-answer-count");
  if (countElement) countElement.textContent = " (" + answers.length + ")";

  var host = document.getElementById("report-answers");
  if (!host) return;
  answers.forEach(function (answer, index) {
    var section = document.createElement("section");
    section.className = "answer";

    var heading = document.createElement("h3");
    var label = answer.model || answer.requested_model || "unknown model";
    heading.textContent = (index + 1) + ". " + label;
    section.appendChild(heading);

    if (answer.error) {
      var failure = document.createElement("p");
      failure.className = "answer-error";
      failure.textContent = "Failed: " + answer.error;
      section.appendChild(failure);
    } else {
      var body = document.createElement("div");
      body.className = "markdown-body";
      renderInto(body, answer.content);
      section.appendChild(body);
    }
    host.appendChild(section);
  });

  if (answers.length === 0) {
    var none = document.createElement("p");
    none.className = "report-meta";
    none.textContent = "No member answers were stored for this run.";
    host.appendChild(none);
  }
})();
"""


def _read_text_asset(path: Path) -> str:
    """Read a bundled UTF-8 text asset, raising StorageError when it is missing."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StorageError(f"Cannot read bundled asset {path}: {exc}") from exc


def _encode_font(name: str) -> str:
    """Return the base64 payload of a vendored woff2 font."""
    try:
        return base64.b64encode((KATEX_FONT_DIR / name).read_bytes()).decode("ascii")
    except OSError as exc:
        raise StorageError(f"Cannot read bundled font {name}: {exc}") from exc


def inline_katex_stylesheet() -> str:
    """Return the KaTeX stylesheet with every font reference inlined as a data URI."""
    stylesheet = _read_text_asset(KATEX_STYLESHEET)
    replaced = 0

    def substitute(match: re.Match[str]) -> str:
        nonlocal replaced
        replaced += 1
        payload = _encode_font(f"{match.group('name')}.woff2")
        return f'src:url(data:{WOFF2_MIME_TYPE};base64,{payload}) format("woff2")'

    inlined = _FONT_FACE_RE.sub(substitute, stylesheet)
    if replaced != _FONT_FACE_COUNT:
        raise StorageError(
            f"Expected {_FONT_FACE_COUNT} KaTeX @font-face rules in {KATEX_STYLESHEET.name} "
            f"but inlined {replaced}; the vendored stylesheet no longer matches the "
            "expected shape, so the report would reference fonts it cannot embed."
        )
    logger.debug("Inlined %d KaTeX font faces as data URIs", replaced)
    return inlined


def _format_timestamp(epoch_seconds: float) -> str:
    """Format Unix seconds for the report, honoring TIME_ZONE then the system locale."""
    if not epoch_seconds or epoch_seconds <= 0:
        return ""
    moment = datetime.fromtimestamp(epoch_seconds, tz=UTC)
    zone_name = os.environ.get("TIME_ZONE")
    if zone_name:
        try:
            moment = moment.astimezone(ZoneInfo(zone_name))
        except (ZoneInfoNotFoundError, ValueError) as exc:
            logger.warning("Ignoring unusable TIME_ZONE %r: %s", zone_name, exc)
            moment = moment.astimezone()
    else:
        moment = moment.astimezone()
    return moment.strftime("%Y-%m-%d %H:%M %Z").strip()


def _report_title(prompt: str | None, fallback: str) -> str:
    """Build a short single-line title from the prompt, truncated on a word boundary."""
    collapsed = " ".join((prompt or "").split())
    if not collapsed:
        return fallback
    if len(collapsed) <= MAX_TITLE_CHARS:
        return collapsed
    clipped = collapsed[: MAX_TITLE_CHARS - 1]
    if " " in clipped:
        clipped = clipped[: clipped.rindex(" ")]
    return clipped.rstrip(" ,;:") + "…"


def _answers_summary(total: int, valid: int) -> str:
    """Describe stored member answers without implying failure when nothing was saved."""
    if total == 0:
        return "no member answers stored"
    return f"{valid} of {total} answers usable"


def _json_payload(payload: Mapping[str, Any]) -> str:
    """Serialize the report payload so it is safe to embed in a script element."""
    encoded = json.dumps(payload, ensure_ascii=False)
    for character, replacement in _JSON_ESCAPES.items():
        encoded = encoded.replace(character, replacement)
    return encoded


def build_report_html(run: Mapping[str, Any]) -> str:
    """Render one stored council run as a fully self-contained HTML document.

    ``run`` uses the shape returned by ``load_history``. The returned document needs
    no network access, but it does need a JavaScript-capable viewer for the Markdown
    and LaTeX to be typeset; a ``<noscript>`` block carries the raw synthesis so the
    file is never blank without it.
    """
    prompt = run.get("prompt")
    synthesis = run.get("synthesis")
    answers = run.get("answers") or []
    judge = run.get("judge") or run.get("judge_requested") or "unknown"
    created = _format_timestamp(float(run.get("created_at") or 0.0))
    title = _report_title(prompt, "Council report")

    valid_answers = sum(1 for answer in answers if answer.get("content"))
    meta_parts = [f"generated {_format_timestamp(datetime.now(UTC).timestamp())}"]
    if created:
        meta_parts.insert(0, f"run {created}")
    meta_parts.append(f"synthesized by {judge}")
    meta_parts.append(_answers_summary(len(answers), valid_answers))
    meta_html = "".join(f"<span>{html_escape(part)}</span>" for part in meta_parts)

    payload = _json_payload(
        {
            "prompt": prompt,
            "synthesis": synthesis,
            "judge": judge,
            "created": created,
            "answers": [
                {
                    "requested_model": answer.get("requested_model"),
                    "model": answer.get("model"),
                    "content": answer.get("content"),
                    "error": answer.get("error"),
                }
                for answer in answers
            ],
        }
    )

    raw_synthesis = synthesis if isinstance(synthesis, str) and synthesis.strip() else ""
    noscript = (
        "<noscript>\n"
        "  <p><strong>JavaScript is disabled.</strong> Showing the raw synthesis "
        "source; enable JavaScript to typeset the Markdown and LaTeX.</p>\n"
        f'  <pre class="prompt-text">{html_escape(raw_synthesis)}</pre>\n'
        "</noscript>"
    )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"<title>{html_escape(title)}</title>\n"
        "<style>\n"
        f"{inline_katex_stylesheet()}\n"
        "</style>\n"
        "<style>\n"
        f"{_REPORT_CSS}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        '<main class="report">\n'
        '  <header class="report-header">\n'
        f"    <h1>{html_escape(title)}</h1>\n"
        f'    <p class="report-meta">{meta_html}</p>\n'
        "  </header>\n"
        "  <section>\n"
        "    <h2>Prompt</h2>\n"
        '    <div class="markdown-body" id="report-prompt"></div>\n'
        '    <details class="raw-source" id="report-prompt-raw"></details>\n'
        "  </section>\n"
        "  <section>\n"
        "    <h2>Council synthesis</h2>\n"
        '    <div class="markdown-body" id="report-synthesis"></div>\n'
        "  </section>\n"
        "  <section>\n"
        '    <h2>Member answers<span id="report-answer-count"></span></h2>\n'
        '    <div id="report-answers"></div>\n'
        "  </section>\n"
        f"  {noscript}\n"
        '  <footer class="report-footer">\n'
        "    Generated by llm-council. Self-contained report: KaTeX stylesheet, fonts, "
        "engine, and the Markdown renderer are all inlined.\n"
        "  </footer>\n"
        "</main>\n"
        f'<script id="report-data" type="application/json">{payload}</script>\n'
        "<script>\n"
        f"{_read_text_asset(KATEX_SCRIPT)}\n"
        "</script>\n"
        "<script>\n"
        f"{_read_text_asset(MARKDOWN_RENDERER)}\n"
        "</script>\n"
        "<script>\n"
        f"{_BOOT_SCRIPT}\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )
