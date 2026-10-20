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

  .usage-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 4px 0 12px;
  }

  .usage-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    background: #f6f8fa;
    border: 1px solid #d8dee4;
    color: #59636e;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 11.5px;
  }

  .usage-pill.provider {
    color: #0969da;
    border-color: rgba(9, 105, 218, 0.3);
    background: rgba(9, 105, 218, 0.06);
  }

  .usage-pill.cost {
    color: #9a6700;
    border-color: rgba(154, 103, 0, 0.35);
    background: rgba(154, 103, 0, 0.06);
  }

  .usage-pill.search {
    color: #8250df;
    border-color: rgba(130, 80, 223, 0.3);
    background: rgba(130, 80, 223, 0.06);
  }

  .citations-strip {
    margin: 12px 0 4px;
    padding: 8px 12px;
    border: 1px solid #e4e8ed;
    border-radius: 6px;
    background: #f6f8fa;
    font-size: 13px;
  }

  .citations-strip summary {
    cursor: pointer;
    color: #59636e;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
    font-weight: 600;
  }

  .citations-list {
    margin: 6px 0 0;
    padding-left: 20px;
  }

  .citations-list li {
    margin: 3px 0;
    line-height: 1.45;
  }

  .citations-list a {
    color: #0969da;
    text-decoration: none;
    word-break: break-all;
  }

  .citations-list a:hover {
    text-decoration: underline;
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

  function extractCitations(usage) {
    if (!usage || typeof usage !== "object") return [];
    var raw = Array.isArray(usage.citations)
      ? usage.citations
      : (Array.isArray(usage.annotations) ? usage.annotations : []);
    var seen = {};
    var out = [];
    for (var i = 0; i < raw.length; i++) {
      var item = raw[i];
      if (!item || typeof item !== "object") continue;
      var nested = item.url_citation && typeof item.url_citation === "object";
      var src = nested ? item.url_citation : item;
      var url = typeof src.url === "string" ? src.url.trim() : "";
      var isHttp = url.indexOf("http://") === 0 || url.indexOf("https://") === 0;
      if (!url || !isHttp || seen[url]) continue;
      seen[url] = true;
      out.push({
        url: url,
        title: typeof src.title === "string" && src.title.trim() ? src.title.trim() : url,
        content: typeof src.content === "string" && src.content.trim() ? src.content.trim() : ""
      });
    }
    return out;
  }

  function formatUsd(val) {
    if (typeof val !== "number" || !isFinite(val) || val <= 0) return "";
    var fixed = val >= 0.01 ? val.toFixed(3) : val.toFixed(4);
    return "$" + fixed.replace(/0+$/, "").replace(/\\.$/, "");
  }

  function buildUsageElement(usage) {
    if (!usage || typeof usage !== "object") return null;
    var orMeta = usage.openrouter_metadata && typeof usage.openrouter_metadata === "object"
      ? usage.openrouter_metadata
      : {};
    // Post-run reconciliation (GET /generation) adds native tokenizer counts and
    // upstream BYOK cost that inline metadata cannot report.
    var recon = usage.generation && typeof usage.generation === "object"
      ? usage.generation
      : {};
    var pills = [];
    function addPill(text, cls, title) {
      var span = document.createElement("span");
      span.className = "usage-pill" + (cls ? " " + cls : "");
      span.textContent = text;
      if (title) span.title = title;
      pills.push(span);
    }

    var provider = typeof usage.provider === "string" ? usage.provider : "";
    if (!provider && Array.isArray(orMeta.endpoints)) {
      for (var i = 0; i < orMeta.endpoints.length; i++) {
        var ep = orMeta.endpoints[i];
        if (ep && ep.selected && (ep.provider || ep.provider_name)) {
          provider = String(ep.provider || ep.provider_name);
          break;
        }
      }
    }
    var attempt = typeof usage.attempt === "number" ? usage.attempt
      : (typeof orMeta.attempt === "number" ? orMeta.attempt : 1);
    var strategy = typeof usage.strategy === "string" ? usage.strategy
      : (typeof orMeta.strategy === "string" ? orMeta.strategy : "");
    var region = typeof usage.region === "string" ? usage.region
      : (typeof orMeta.region === "string" ? orMeta.region : "");
    if (provider) {
      var attemptSuffix = attempt > 1 ? " · attempt " + attempt : "";
      var tooltipParts = ["Provider: " + provider];
      if (strategy) tooltipParts.push("strategy: " + strategy);
      if (region) tooltipParts.push("region: " + region);
      if (attempt > 1) tooltipParts.push("attempt: " + attempt);
      addPill("via " + provider + attemptSuffix, "provider", tooltipParts.join(" · "));
    } else if (attempt > 1) {
      addPill("attempt " + attempt, "provider", "");
    }

    var genMs = typeof usage.generation_time === "number" ? usage.generation_time
      : (typeof orMeta.generation_time === "number" ? orMeta.generation_time : null);
    var elapsedMs = typeof usage.elapsed_ms === "number" ? usage.elapsed_ms : null;
    var displayMs = genMs !== null && genMs > 0 ? genMs : elapsedMs;
    if (typeof displayMs === "number" && displayMs >= 0) {
      var secText = displayMs >= 1000
        ? (displayMs / 1000).toFixed(1) + "s"
        : Math.round(displayMs) + "ms";
      var genLabel = genMs !== null ? "Generation: " + Math.round(genMs) + "ms" : "";
      var timingTitle = genMs !== null && elapsedMs !== null
        ? genLabel + " · Wall-clock: " + Math.round(elapsedMs) + "ms"
        : (genLabel || "Latency: " + Math.round(displayMs) + "ms");
      addPill("⏱ " + secText, "", timingTitle);
    }

    var totalTok = typeof usage.total_tokens === "number" ? usage.total_tokens : null;
    var promptTok = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : null;
    var compTok = typeof usage.completion_tokens === "number" ? usage.completion_tokens : null;
    var compRaw = usage.completion_tokens_details;
    var compDetails = compRaw && typeof compRaw === "object" ? compRaw : {};
    var promptRaw = usage.prompt_tokens_details;
    var promptDetails = promptRaw && typeof promptRaw === "object" ? promptRaw : {};
    var rVal = compDetails.reasoning_tokens;
    var reasoningTok = typeof rVal === "number" && rVal > 0 ? rVal : 0;
    var cVal = promptDetails.cached_tokens;
    var cachedTok = typeof cVal === "number" && cVal > 0 ? cVal : 0;

    if (totalTok !== null && totalTok > 0) {
      var details = [];
      if (reasoningTok > 0 && compTok !== null) {
        var visibleTok = Math.max(compTok - reasoningTok, 0);
        details.push(visibleTok + " ans + " + reasoningTok + " reasoning");
      } else if (reasoningTok > 0) {
        details.push(reasoningTok + " reasoning");
      }
      if (cachedTok > 0) details.push(cachedTok + " cached");
      var label = details.length > 0
        ? totalTok + " tok (" + details.join(" · ") + ")"
        : totalTok + " tok";
      var tokTitle = [];
      if (promptTok !== null) tokTitle.push("Prompt: " + promptTok);
      if (compTok !== null) tokTitle.push("Completion: " + compTok);
      if (reasoningTok > 0) tokTitle.push("Reasoning: " + reasoningTok);
      if (cachedTok > 0) tokTitle.push("Cached read: " + cachedTok);
      if (typeof recon.native_tokens_prompt === "number") {
        tokTitle.push("Native prompt: " + recon.native_tokens_prompt);
      }
      if (typeof recon.native_tokens_completion === "number") {
        tokTitle.push("Native completion: " + recon.native_tokens_completion);
      }
      addPill(label, "", tokTitle.join(" · "));
    }

    var toolUse = usage.server_tool_use && typeof usage.server_tool_use === "object"
      ? usage.server_tool_use
      : {};
    var sReqs = toolUse.web_search_requests;
    var searchReqs = typeof sReqs === "number" && sReqs > 0 ? sReqs : 0;
    var costDetails = usage.cost_details && typeof usage.cost_details === "object"
      ? usage.cost_details
      : {};
    var sCost = costDetails.server_tool_cost;
    var toolCost = typeof sCost === "number" && sCost > 0 ? sCost : 0;
    var totalCost = typeof usage.cost === "number" && usage.cost > 0 ? usage.cost : 0;

    if (searchReqs > 0) {
      var searchLabel = searchReqs + " search" + (searchReqs === 1 ? "" : "es")
        + (toolCost > 0 ? " (" + formatUsd(toolCost) + ")" : "");
      addPill("🌐 " + searchLabel, "search", "OpenRouter server-side web search");
    }
    var rawDiscount = recon.cache_discount;
    var cacheDiscount = typeof rawDiscount === "number" && rawDiscount > 0 ? rawDiscount : 0;
    if (totalCost > 0) {
      var costSuffix = toolCost > 0
        ? " (includes " + formatUsd(toolCost) + " web search)"
        : "";
      if (cacheDiscount > 0) {
        costSuffix += " · cache discount " + formatUsd(cacheDiscount);
      }
      var costTitle = toolCost > 0
        ? "Total cost: " + formatUsd(totalCost) + costSuffix
        : "Cost: " + formatUsd(totalCost) + costSuffix;
      addPill(formatUsd(totalCost), "cost", costTitle);
    } else if (toolCost > 0 && searchReqs === 0) {
      addPill(formatUsd(toolCost), "cost", "Web search cost");
    }

    var rawUpstream = recon.upstream_inference_cost;
    var upstreamCost = typeof rawUpstream === "number" && rawUpstream > 0 ? rawUpstream : 0;
    if (upstreamCost > 0) {
      addPill(
        "upstream " + formatUsd(upstreamCost),
        "cost",
        "Upstream provider cost (BYOK): " + formatUsd(upstreamCost)
          + " · reconciled from GET /generation"
      );
    }

    var rawGenId = usage.generation_id;
    var genId = typeof rawGenId === "string" && rawGenId ? rawGenId : "";
    var nativeStop = typeof usage.native_finish_reason === "string" && usage.native_finish_reason
      ? usage.native_finish_reason
      : "";
    if (genId) {
      var shortGen = genId.length > 16 ? genId.slice(0, 14) + "…" : genId;
      var genTitle = nativeStop
        ? "Generation ID: " + genId + " · stop: " + nativeStop
        : "Generation ID: " + genId;
      addPill(shortGen, "", genTitle);
    }

    if (pills.length === 0) return null;
    var row = document.createElement("div");
    row.className = "usage-pills";
    for (var j = 0; j < pills.length; j++) row.appendChild(pills[j]);
    return row;
  }

  function buildCitationsElement(usage) {
    var citations = extractCitations(usage);
    if (citations.length === 0) return null;
    var detailsEl = document.createElement("details");
    detailsEl.className = "citations-strip";
    var summaryEl = document.createElement("summary");
    var plural = citations.length === 1 ? "" : "s";
    summaryEl.textContent = "🔗 " + citations.length + " web citation" + plural;
    detailsEl.appendChild(summaryEl);
    var listEl = document.createElement("ol");
    listEl.className = "citations-list";
    for (var i = 0; i < citations.length; i++) {
      var c = citations[i];
      var li = document.createElement("li");
      var link = document.createElement("a");
      link.href = c.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = c.title;
      li.appendChild(link);
      if (c.content) {
        var snippet = document.createElement("span");
        var clipped = c.content.length > 140 ? c.content.slice(0, 140) + "…" : c.content;
        snippet.textContent = " — " + clipped;
        li.appendChild(snippet);
      }
      listEl.appendChild(li);
    }
    detailsEl.appendChild(listEl);
    return detailsEl;
  }

  var synthMetaHost = document.getElementById("report-synthesis-meta");
  if (synthMetaHost) {
    var synthPills = buildUsageElement(data.synthesis_usage);
    if (synthPills) synthMetaHost.appendChild(synthPills);
  }
  var synthCitationsHost = document.getElementById("report-synthesis-citations");
  if (synthCitationsHost) {
    var synthCites = buildCitationsElement(data.synthesis_usage);
    if (synthCites) synthCitationsHost.appendChild(synthCites);
  }

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
    if (answer.requested_model && answer.model && answer.requested_model !== answer.model) {
      label += " (via " + answer.requested_model + ")";
    }
    heading.textContent = (index + 1) + ". " + label;
    section.appendChild(heading);

    var memberPills = buildUsageElement(answer.usage);
    if (memberPills) section.appendChild(memberPills);

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
      var memberCites = buildCitationsElement(answer.usage);
      if (memberCites) section.appendChild(memberCites);
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
            "synthesis_usage": run.get("synthesis_usage") or {},
            "judge": judge,
            "created": created,
            "answers": [
                {
                    "requested_model": answer.get("requested_model"),
                    "model": answer.get("model"),
                    "content": answer.get("content"),
                    "error": answer.get("error"),
                    "usage": answer.get("usage") or {},
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
        '    <div id="report-synthesis-meta"></div>\n'
        '    <div class="markdown-body" id="report-synthesis"></div>\n'
        '    <div id="report-synthesis-citations"></div>\n'
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
