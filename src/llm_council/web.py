"""Interactive local web UI and HTTP/SSE server for llm-council."""
# ruff: noqa: E501

from __future__ import annotations

import json
import logging
import os
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event
from typing import Any
from urllib.parse import unquote, urlparse

from .config import Settings
from .council import improve_prompt, query_result
from .errors import CouncilError
from .judges import STRATEGIES
from .storage import load_history

logger = logging.getLogger("llm_council")

DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
MAX_REQUEST_BYTES = 10 * 1024 * 1024

# Vendored browser assets (KaTeX, used to typeset LaTeX in council answers).
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_URL_PREFIX = "/static/"
STATIC_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
}

_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>llm-council — Interactive Workspace</title>
<link rel="stylesheet" href="/static/katex/katex.min.css" />
<script src="/static/katex/katex.min.js"></script>
<script src="/static/markdown_renderer.js"></script>
<style>
  :root {
    --bg: #0b0e14;
    --bg-panel: #111621;
    --bg-elev: #161d2b;
    --bg-hover: #1d2638;
    --bg-input: #0d121b;
    --border: #232d42;
    --border-strong: #334363;
    --text: #e6edf3;
    --text-muted: #93a1b8;
    --text-dim: #63718a;
    --accent: #58a6ff;
    --accent-strong: #2f81f7;
    --accent-soft: rgba(88, 166, 255, 0.14);
    --green: #3fb950;
    --green-soft: rgba(63, 185, 80, 0.14);
    --amber: #d29922;
    --amber-soft: rgba(210, 153, 34, 0.15);
    --red: #f85149;
    --red-soft: rgba(248, 81, 73, 0.14);
    --purple: #bc8cff;
    --purple-soft: rgba(188, 140, 255, 0.14);
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.55;
  }

  .app-shell {
    display: grid;
    grid-template-columns: 300px 1fr;
    height: 100vh;
    overflow: hidden;
  }

  /* Left-hand History Sidebar */
  .sidebar {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  .sidebar-header {
    padding: 16px 16px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 9px;
    font-family: var(--mono);
    font-weight: 700;
    font-size: 15px;
    color: var(--text);
  }

  .brand-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
  }

  .sidebar-search {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }

  .sidebar-search input {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 6px 10px;
    font-size: 12.5px;
    outline: none;
  }

  .sidebar-search input:focus {
    border-color: var(--accent);
  }

  .history-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 10px 10px 18px;
  }

  .history-section-title {
    font-family: var(--mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    padding: 8px 8px 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .history-section-title.db-title {
    color: var(--text-dim);
    margin-top: 12px;
    border-top: 1px solid var(--border);
    padding-top: 14px;
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .history-item {
    width: 100%;
    text-align: left;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 11px;
    color: var(--text);
    cursor: pointer;
    transition: background 0.14s ease, border-color 0.14s ease, opacity 0.14s ease;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .history-item:hover {
    background: var(--bg-hover);
    border-color: var(--border-strong);
  }

  .history-item.active {
    border-color: var(--accent);
    background: var(--accent-soft);
  }

  /* DB items are rendered less prominently than current session items */
  .history-item.db-item {
    background: rgba(22, 29, 43, 0.45);
    border-color: rgba(35, 45, 66, 0.65);
    opacity: 0.72;
    padding: 7px 10px;
  }

  .history-item.db-item:hover {
    opacity: 0.95;
    background: var(--bg-elev);
  }

  .history-item.db-item.active {
    opacity: 1;
    border-color: var(--accent);
    background: var(--accent-soft);
  }

  .history-prompt {
    font-size: 12.8px;
    font-weight: 500;
    color: var(--text);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    word-break: break-word;
  }

  .history-item.db-item .history-prompt {
    font-size: 12px;
    font-weight: 400;
    color: var(--text-muted);
  }

  .history-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--text-dim);
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 1px 6px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 10px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-muted);
  }

  .badge.session {
    color: var(--accent);
    border-color: rgba(88, 166, 255, 0.35);
    background: var(--accent-soft);
  }

  .badge.followup {
    color: var(--purple);
    border-color: rgba(188, 140, 255, 0.35);
    background: var(--purple-soft);
  }

  .badge.synth {
    color: var(--green);
    border-color: rgba(63, 185, 80, 0.35);
    background: var(--green-soft);
  }

  .empty-history {
    padding: 12px 8px;
    font-size: 12px;
    color: var(--text-dim);
    font-style: italic;
  }

  /* Main Area */
  .main-pane {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* Composer Section */
  .composer-card {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .composer-top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .composer-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
  }

  .selected-context-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 9px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 11px;
    background: var(--purple-soft);
    color: var(--purple);
    border: 1px solid rgba(188, 140, 255, 0.35);
  }

  .prompt-textarea {
    width: 100%;
    min-height: 98px;
    max-height: 240px;
    resize: vertical;
    background: var(--bg-input);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    padding: 12px 14px;
    font-family: var(--sans);
    font-size: 14.5px;
    line-height: 1.5;
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }

  .prompt-textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.18);
  }

  .attachment-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .attachment-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 10px;
    border-radius: 6px;
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--accent);
  }

  .attachment-chip button {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 13px;
    padding: 0 2px;
  }

  .attachment-chip button:hover {
    color: var(--red);
  }

  .composer-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .controls-left, .controls-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    background: var(--bg-elev);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 7px 11px;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.14s ease, border-color 0.14s ease;
  }

  .icon-btn:hover {
    background: var(--bg-hover);
    border-color: var(--accent);
  }

  .judge-group {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 4px 9px;
  }

  .judge-group label {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text-muted);
  }

  .search-web-toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
    padding: 6px 9px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-input);
    white-space: nowrap;
  }

  .search-web-toggle:hover {
    border-color: var(--border-strong);
    color: var(--text);
  }

  .search-web-toggle input {
    accent-color: var(--accent-strong);
    cursor: pointer;
    margin: 0;
  }

  .search-web-toggle.enabled {
    color: var(--text);
    border-color: var(--accent-strong);
    background: var(--accent-soft);
  }

  .judge-select, .judge-custom-input {
    background: var(--bg-input);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
    font-family: var(--mono);
    font-size: 12px;
    outline: none;
  }

  .judge-select:focus, .judge-custom-input:focus {
    border-color: var(--accent);
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    padding: 8px 15px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13.5px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.14s ease;
    position: relative;
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .btn-primary {
    background: var(--accent-strong);
    color: #ffffff;
    border-color: rgba(255, 255, 255, 0.12);
  }

  .btn-primary:hover:not(:disabled) {
    background: var(--accent);
  }

  .btn-followup {
    background: var(--purple-soft);
    color: var(--purple);
    border-color: rgba(188, 140, 255, 0.45);
  }

  .btn-followup:hover:not(:disabled) {
    background: rgba(188, 140, 255, 0.24);
    border-color: var(--purple);
  }

  .btn-secondary {
    background: var(--bg-elev);
    color: var(--text);
    border-color: var(--border-strong);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--bg-hover);
    border-color: var(--accent);
  }

  .btn-danger {
    background: var(--red-soft);
    color: var(--red);
    border-color: rgba(248, 81, 73, 0.55);
  }

  .btn-danger:hover:not(:disabled) {
    background: rgba(248, 81, 73, 0.26);
    border-color: var(--red);
    color: #ff948f;
  }

  .btn-danger:disabled {
    background: var(--bg-elev);
    color: var(--text-dim);
    border-color: var(--border);
    opacity: 0.5;
  }

  /* Rich hover tooltips for <New> and <Followup> */
  [data-tooltip] {
    position: relative;
  }

  [data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    width: max-content;
    max-width: 310px;
    background: #070a0f;
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 7px 10px;
    font-family: var(--sans);
    font-weight: 400;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);
    opacity: 0;
    pointer-events: none;
    transform: translateY(-4px);
    transition: opacity 0.14s ease, transform 0.14s ease;
    z-index: 40;
    white-space: normal;
    text-align: left;
  }

  .controls-left [data-tooltip]::after {
    left: 0;
    right: auto;
  }

  [data-tooltip]:hover::after,
  [data-tooltip]:focus-visible::after {
    opacity: 1;
    transform: translateY(0);
  }

  /* Improved prompts suggestion drawer */
  .improve-drawer {
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 10px 12px;
    display: none;
    flex-direction: column;
    gap: 8px;
  }

  .improve-drawer.open {
    display: flex;
  }

  .improve-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--accent);
  }

  .improve-options {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .improve-option-btn {
    text-align: left;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 10px;
    color: var(--text);
    font-size: 12.5px;
    cursor: pointer;
  }

  .improve-option-btn:hover {
    border-color: var(--accent);
    background: var(--bg-hover);
  }

  /* Workspace / Answers area */
  .results-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 18px 24px 36px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .status-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 9px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 12px;
  }

  .status-banner.running {
    border-color: rgba(88, 166, 255, 0.45);
    background: var(--accent-soft);
  }

  .status-banner.error {
    border-color: rgba(248, 81, 73, 0.45);
    background: var(--red-soft);
    color: #ff948f;
  }

  .status-banner.warning {
    border-color: rgba(210, 153, 34, 0.45);
    background: var(--amber-soft);
    color: #f2cc60;
  }

  /* Pagination Bar (shown once synthesis appears or when browsing answers) */
  .pagination-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
  }

  .page-tabs {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .page-tab {
    background: var(--bg-elev);
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 6px 11px;
    font-family: var(--mono);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.14s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .page-tab:hover {
    background: var(--bg-hover);
    color: var(--text);
  }

  .page-tab.active {
    background: var(--accent-soft);
    color: var(--accent);
    border-color: var(--accent);
    font-weight: 600;
  }

  .page-tab.synthesis-tab.active {
    background: var(--green-soft);
    color: var(--green);
    border-color: var(--green);
  }

  .page-tab.failed-tab {
    color: var(--red);
    border-color: rgba(248, 81, 73, 0.35);
  }

  .pager-nav {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .pager-btn {
    background: var(--bg-elev);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    padding: 5px 10px;
    font-family: var(--mono);
    font-size: 12px;
    cursor: pointer;
  }

  .pager-btn:hover:not(:disabled) {
    border-color: var(--accent);
    background: var(--bg-hover);
  }

  .pager-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Answer Cards */
  .answers-stream {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .answer-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
  }

  .answer-card.synthesis-card {
    border-color: rgba(63, 185, 80, 0.5);
  }

  .answer-card.error-card {
    border-color: rgba(248, 81, 73, 0.45);
  }

  .answer-header {
    padding: 11px 16px;
    background: var(--bg-elev);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
  }

  .answer-model {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
  }

  .answer-meta {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
  }

  .meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 6px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-family: var(--mono);
    font-size: 10.5px;
    white-space: nowrap;
  }

  .meta-pill.provider {
    color: var(--accent);
    border-color: rgba(88, 166, 255, 0.35);
    background: var(--accent-soft);
  }

  .meta-pill.cost {
    color: var(--amber);
    border-color: rgba(210, 153, 34, 0.35);
    background: var(--amber-soft);
  }

  .meta-pill.search {
    color: var(--purple);
    border-color: rgba(188, 140, 255, 0.35);
    background: var(--purple-soft);
  }

  .citations-strip {
    margin: 0 20px 14px;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(88, 166, 255, 0.06);
    border: 1px solid rgba(88, 166, 255, 0.25);
    font-size: 12px;
  }

  .citations-strip summary {
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    user-select: none;
  }

  .citations-list {
    margin: 8px 0 0;
    padding-left: 18px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .citations-list li {
    color: var(--text-muted);
    line-height: 1.4;
  }

  .citations-list a {
    color: var(--accent);
    text-decoration: none;
    word-break: break-all;
  }

  .citations-list a:hover {
    text-decoration: underline;
  }

  .copy-btn {
    background: var(--bg-panel);
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 8px;
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
  }

  .copy-btn:hover {
    color: var(--text);
    border-color: var(--accent);
  }

  /* Rendered Markdown Body */
  .markdown-body {
    padding: 16px 20px;
    color: var(--text);
    font-size: 14.5px;
    line-height: 1.65;
    overflow-wrap: break-word;
  }

  .markdown-body h1,
  .markdown-body h2,
  .markdown-body h3,
  .markdown-body h4 {
    margin: 16px 0 8px;
    font-weight: 650;
    line-height: 1.3;
    color: #ffffff;
  }

  .markdown-body h1:first-child,
  .markdown-body h2:first-child,
  .markdown-body h3:first-child,
  .markdown-body p:first-child {
    margin-top: 0;
  }

  .markdown-body p {
    margin: 0 0 12px;
  }

  .markdown-body p:last-child {
    margin-bottom: 0;
  }

  .markdown-body ul, .markdown-body ol {
    margin: 0 0 12px;
    padding-left: 22px;
  }

  .markdown-body li {
    margin-bottom: 4px;
  }

  .markdown-body code {
    font-family: var(--mono);
    font-size: 12.5px;
    background: rgba(110, 118, 129, 0.22);
    padding: 2px 5px;
    border-radius: 4px;
  }

  .markdown-body pre {
    background: #080b10;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    overflow-x: auto;
    margin: 10px 0 14px;
  }

  .markdown-body pre code {
    background: none;
    padding: 0;
    font-size: 12.5px;
    color: #d2dce8;
  }

  .markdown-body blockquote {
    margin: 10px 0 14px;
    padding: 6px 14px;
    border-left: 3px solid var(--accent);
    background: rgba(88, 166, 255, 0.07);
    color: var(--text-muted);
  }

  .markdown-body table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 13px;
  }

  .markdown-body th, .markdown-body td {
    border: 1px solid var(--border);
    padding: 7px 10px;
    text-align: left;
  }

  .markdown-body th {
    background: var(--bg-elev);
    font-weight: 600;
  }

  /* KaTeX typesetting: nudge the copy size up slightly and let wide
     display equations scroll horizontally instead of overflowing. */
  .markdown-body .katex {
    font-size: 1.08em;
  }

  .math-display {
    overflow-x: auto;
    overflow-y: hidden;
    padding: 2px 0;
  }

  .math-display .katex-display {
    margin: 0.7em 0;
  }

  .math-raw {
    color: var(--text-muted);
  }

  /* Literal source kept beside a rendered document view, so a rewritten prompt can
     always be compared against exactly what was stored. */
  .raw-source {
    margin-top: 12px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
  }

  .raw-source summary {
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text-muted);
  }

  .raw-source pre {
    margin: 8px 0 0;
    padding: 10px 12px;
    background: #080b10;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    color: #d2dce8;
    font-size: 12px;
  }

  .empty-workspace {
    margin: auto;
    max-width: 520px;
    text-align: center;
    padding: 42px 24px;
    color: var(--text-muted);
  }

  .empty-workspace h2 {
    color: var(--text);
    font-size: 18px;
    margin: 0 0 8px;
  }

  @media (max-width: 860px) {
    .app-shell {
      grid-template-columns: 1fr;
    }
    .sidebar {
      height: 220px;
      border-right: none;
      border-bottom: 1px solid var(--border);
    }
  }
</style>
</head>
<body>
<div class="app-shell">
  <!-- Left-Hand History Panel -->
  <aside class="sidebar" aria-label="Conversation History">
    <div class="sidebar-header">
      <div class="brand">
        <span class="brand-dot"></span>
        <span>llm-council</span>
      </div>
      <span id="history-count-badge" class="badge">0 runs</span>
    </div>
    <div class="sidebar-search">
      <input
        id="history-filter-input"
        type="search"
        placeholder="Filter history prompts..."
        aria-label="Filter history prompts"
      />
    </div>
    <div class="history-scroll">
      <div class="history-section-title">
        <span>Current Session</span>
        <span id="session-count-label">0</span>
      </div>
      <div id="session-history-list" class="history-list" role="list"></div>

      <div class="history-section-title db-title">
        <span>Saved in DuckDB</span>
        <span id="db-count-label">0</span>
      </div>
      <div id="db-history-list" class="history-list" role="list"></div>
    </div>
  </aside>

  <!-- Main Workspace -->
  <main class="main-pane">
    <!-- Prompt Composer -->
    <section class="composer-card" aria-label="Prompt Composer">
      <div class="composer-top-bar">
        <div class="composer-title">
          <span>Council Prompt</span>
          <span id="active-context-pill" class="selected-context-pill" style="display: none;"></span>
        </div>
        <div style="font-family: var(--mono); font-size: 11.5px; color: var(--text-dim);">
          Ctrl+Enter for New · Shift+Ctrl+Enter for Followup
        </div>
      </div>

      <textarea
        id="prompt-input"
        class="prompt-textarea"
        placeholder="Ask the free OpenRouter model council a question, or type a follow-up to the selected conversation..."
        aria-label="Prompt text"
      ></textarea>

      <div id="attachment-row" class="attachment-row" style="display: none;"></div>

      <div id="improve-drawer" class="improve-drawer" aria-live="polite">
        <div class="improve-header">
          <span id="improve-header-title">Suggested Prompt Improvements (click to use)</span>
          <button id="close-improve-btn" class="copy-btn" type="button">Close</button>
        </div>
        <div id="improve-options" class="improve-options"></div>
      </div>

      <div class="composer-controls">
        <div class="controls-left">
          <input id="file-input" type="file" style="display: none;" />
          <button
            id="attach-file-btn"
            type="button"
            class="icon-btn"
            title="Attach a text/markdown file to include with your prompt"
            aria-label="Attach file"
          >
            <span aria-hidden="true">📎</span>
            <span>Attach file</span>
          </button>

          <div class="judge-group">
            <label for="judge-select">Judge criterion:</label>
            <select id="judge-select" class="judge-select" aria-label="Judge criterion">
              <option value="random">random (random council member)</option>
              <option value="round-robin">round-robin (rotate across models)</option>
              <option value="most-intelligent">most-intelligent (OpenRouter ranking)</option>
              <option value="largest-context">largest-context (max context window)</option>
              <option value="free-router">free-router (FREE_ROUTER_MODEL)</option>
              <option value="auto">auto (AUTO_ROUTER_MODEL)</option>
              <option value="custom">explicit model slug...</option>
            </select>
            <input
              id="judge-custom-input"
              type="text"
              class="judge-custom-input"
              placeholder="vendor/model:free"
              aria-label="Custom judge model slug"
              style="display: none; width: 170px;"
            />
          </div>

          <label
            id="search-web-toggle"
            class="search-web-toggle"
            for="search-web-checkbox"
            title="Let OpenRouter run server-side web searches for tool-capable council members and the judge. Searches are billed per request on top of token costs, even for free models."
          >
            <input id="search-web-checkbox" type="checkbox" />
            <span>🌐 Web search</span>
          </label>

          <button
            id="improve-btn"
            type="button"
            class="btn btn-secondary"
            data-tooltip="Ask a single LLM to propose 3–5 sharper, more structured versions of your current prompt before querying the full council."
          >
            ✨ Improve Prompt
          </button>
        </div>

        <div class="controls-right">
          <button
            id="clear-btn"
            type="button"
            class="btn btn-secondary"
            data-tooltip="Clear the prompt textbox so you can type a fresh question."
          >
            ✕ Clear
          </button>
          <button
            id="stop-btn"
            type="button"
            class="btn btn-danger"
            disabled
            data-tooltip="Stop / Abort the active LLM retrieval in progress."
          >
            ■ Stop
          </button>
          <button
            id="followup-btn"
            type="button"
            class="btn btn-followup"
            disabled
            data-tooltip="Send a follow-up query that attaches the selected conversation's prompt and synthesis as prior context."
          >
            ↩ Followup
          </button>
          <button
            id="new-btn"
            type="button"
            class="btn btn-primary"
            data-tooltip="Start a brand-new council run with no prior conversation context attached."
          >
            ✦ New
          </button>
        </div>
      </div>
    </section>

    <!-- Main Answer Rendering Area -->
    <section class="results-scroll" id="results-area" aria-live="polite">
      <div id="status-banner" class="status-banner" style="display: none;"></div>
      <div id="pagination-bar" class="pagination-bar" style="display: none;"></div>
      <div id="answers-container" class="answers-stream">
        <div class="empty-workspace" id="empty-workspace">
          <h2>Ready to convene the council</h2>
          <p>
            Enter a prompt above and click <strong>New</strong> to query all free OpenRouter models in parallel,
            or select a conversation from the left-hand history panel to inspect its answers or send a <strong>Followup</strong>.
          </p>
        </div>
      </div>
    </section>
  </main>
</div>

<script>
(function() {
  const state = {
    sessionRuns: [],
    dbRuns: [],
    selectedRunId: null,
    attachedFile: null, // { name, content }
    isRunning: false,
    activeAbortController: null,
    activePage: "synthesis", // "synthesis" | "all" | number (0-based answer index)
    filterText: "",
  };

  const promptInput = document.getElementById("prompt-input");
  const fileInput = document.getElementById("file-input");
  const attachFileBtn = document.getElementById("attach-file-btn");
  const attachmentRow = document.getElementById("attachment-row");
  const judgeSelect = document.getElementById("judge-select");
  const searchWebToggle = document.getElementById("search-web-toggle");
  const searchWebCheckbox = document.getElementById("search-web-checkbox");
  const judgeCustomInput = document.getElementById("judge-custom-input");
  const clearBtn = document.getElementById("clear-btn");
  const stopBtn = document.getElementById("stop-btn");
  const newBtn = document.getElementById("new-btn");
  const followupBtn = document.getElementById("followup-btn");
  const improveBtn = document.getElementById("improve-btn");
  const improveDrawer = document.getElementById("improve-drawer");
  const improveOptions = document.getElementById("improve-options");
  const improveHeaderTitle = document.getElementById("improve-header-title");
  const closeImproveBtn = document.getElementById("close-improve-btn");
  const sessionListEl = document.getElementById("session-history-list");
  const dbListEl = document.getElementById("db-history-list");
  const sessionCountLabel = document.getElementById("session-count-label");
  const dbCountLabel = document.getElementById("db-count-label");
  const historyCountBadge = document.getElementById("history-count-badge");
  const historyFilterInput = document.getElementById("history-filter-input");
  const statusBanner = document.getElementById("status-banner");
  const paginationBar = document.getElementById("pagination-bar");
  const answersContainer = document.getElementById("answers-container");
  const activeContextPill = document.getElementById("active-context-pill");

  function findRunById(id) {
    if (!id) return null;
    return state.sessionRuns.find(r => r.id === id) || state.dbRuns.find(r => r.id === id) || null;
  }

  function getSelectedJudgeValue() {
    if (judgeSelect.value === "custom") {
      return judgeCustomInput.value.trim() || "random";
    }
    return judgeSelect.value;
  }

  function formatTimestamp(epochSeconds) {
    if (!epochSeconds) return "";
    const d = new Date(epochSeconds * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function updateFollowupButtonState() {
    const selected = findRunById(state.selectedRunId);
    const canFollowup = Boolean(selected && selected.synthesis && !state.isRunning);
    followupBtn.disabled = !canFollowup;
    stopBtn.disabled = !state.isRunning;

    if (selected && selected.synthesis) {
      const snippet = (selected.prompt || "Selected run").slice(0, 44);
      activeContextPill.style.display = "inline-flex";
      activeContextPill.textContent = "Selected: " + snippet + (selected.prompt && selected.prompt.length > 44 ? "…" : "");
      followupBtn.setAttribute(
        "data-tooltip",
        "Follow up on selected conversation (\"" + snippet.replace(/"/g, "'") + "\"): attaches its synthesis and prior turns as context."
      );
    } else if (selected) {
      activeContextPill.style.display = "inline-flex";
      activeContextPill.textContent = "Selected (no synthesis saved)";
      followupBtn.setAttribute(
        "data-tooltip",
        "This historical run has no saved synthesis, so Followup is unavailable. Click New to run a fresh council query."
      );
    } else {
      activeContextPill.style.display = "none";
      followupBtn.setAttribute(
        "data-tooltip",
        "Select or complete a conversation with a synthesis to attach its synthesis as follow-up context."
      );
    }
  }

  function renderAttachmentChip() {
    if (!state.attachedFile) {
      attachmentRow.style.display = "none";
      attachmentRow.innerHTML = "";
      return;
    }
    attachmentRow.style.display = "flex";
    const sizeKb = (state.attachedFile.content.length / 1024).toFixed(1);
    attachmentRow.innerHTML = `
      <div class="attachment-chip">
        <span>📎 ${escapeHtml(state.attachedFile.name)} (${sizeKb} KB)</span>
        <button type="button" id="remove-attachment-btn" title="Remove attached file" aria-label="Remove attached file">✕</button>
      </div>
    `;
    const removeBtn = document.getElementById("remove-attachment-btn");
    if (removeBtn) {
      removeBtn.addEventListener("click", function() {
        state.attachedFile = null;
        fileInput.value = "";
        renderAttachmentChip();
      });
    }
  }

  function renderHistorySidebar() {
    const q = state.filterText.trim().toLowerCase();
    const sessionIds = new Set(state.sessionRuns.map(r => r.id));
    const filteredSession = state.sessionRuns.filter(r => !q || (r.prompt || "").toLowerCase().includes(q));
    const dedupedDb = state.dbRuns.filter(r => !sessionIds.has(r.id));
    const filteredDb = dedupedDb.filter(r => !q || (r.prompt || "").toLowerCase().includes(q));

    sessionCountLabel.textContent = String(filteredSession.length);
    dbCountLabel.textContent = String(filteredDb.length);
    historyCountBadge.textContent = (state.sessionRuns.length + dedupedDb.length) + " runs";

    if (filteredSession.length === 0) {
      sessionListEl.innerHTML = '<div class="empty-history">No prompts in this session yet.</div>';
    } else {
      sessionListEl.innerHTML = filteredSession.map(run => {
        const active = run.id === state.selectedRunId ? "active" : "";
        const promptLabel = run.prompt || "(Prompt not saved)";
        const ansCount = (run.answers || []).length;
        const synthBadge = run.synthesis
          ? '<span class="badge synth">✓ synthesis</span>'
          : (run.running ? '<span class="badge session">streaming…</span>' : '');
        const followupBadge = run.isFollowup ? '<span class="badge followup">↩ followup</span>' : '';
        return `
          <button type="button" class="history-item ${active}" data-run-id="${escapeHtml(run.id)}">
            <div class="history-prompt">${escapeHtml(promptLabel)}</div>
            <div class="history-meta">
              <span>${ansCount} ans · ${escapeHtml(formatTimestamp(run.created_at))}</span>
              <span>${followupBadge} ${synthBadge}</span>
            </div>
          </button>
        `;
      }).join("");
    }

    if (filteredDb.length === 0) {
      dbListEl.innerHTML = '<div class="empty-history">No stored DuckDB runs found.</div>';
    } else {
      dbListEl.innerHTML = filteredDb.map(run => {
        const active = run.id === state.selectedRunId ? "active" : "";
        const promptLabel = run.prompt || "(Prompt text not saved in DB)";
        const ansCount = (run.answers || []).length;
        const avail = run.synthesis
          ? "synth" + (ansCount ? ` + ${ansCount} ans` : "")
          : (ansCount ? `${ansCount} ans only` : "prompt only");
        return `
          <button type="button" class="history-item db-item ${active}" data-run-id="${escapeHtml(run.id)}">
            <div class="history-prompt">${escapeHtml(promptLabel)}</div>
            <div class="history-meta">
              <span>${escapeHtml(avail)}</span>
              <span>${escapeHtml(formatTimestamp(run.created_at))}</span>
            </div>
          </button>
        `;
      }).join("");
    }

    document.querySelectorAll(".history-item[data-run-id]").forEach(btn => {
      btn.addEventListener("click", function() {
        const runId = btn.getAttribute("data-run-id");
        selectRun(runId, { populatePrompt: true });
      });
    });
  }

  function selectRun(runId, opts) {
    const options = opts || {};
    state.selectedRunId = runId;
    const run = findRunById(runId);
    if (run) {
      if (options.populatePrompt && run.prompt) {
        promptInput.value = run.prompt;
      }
      if (run.synthesis) {
        state.activePage = "synthesis";
      } else if (run.answers && run.answers.length > 0) {
        state.activePage = 0;
      } else {
        state.activePage = "all";
      }
    }
    updateFollowupButtonState();
    renderHistorySidebar();
    renderWorkspace();
  }

  function extractUsageCitations(usage) {
    if (!usage || typeof usage !== "object") return [];
    const raw = Array.isArray(usage.citations)
      ? usage.citations
      : (Array.isArray(usage.annotations) ? usage.annotations : []);
    const seen = new Set();
    const out = [];
    for (const item of raw) {
      if (!item || typeof item !== "object") continue;
      const src = item.url_citation && typeof item.url_citation === "object" ? item.url_citation : item;
      const url = typeof src.url === "string" ? src.url.trim() : "";
      if (!url || (!url.startsWith("http://") && !url.startsWith("https://")) || seen.has(url)) continue;
      seen.add(url);
      out.push({
        url,
        title: typeof src.title === "string" && src.title.trim() ? src.title.trim() : url,
        content: typeof src.content === "string" && src.content.trim() ? src.content.trim() : "",
      });
    }
    return out;
  }

  function formatUsd(val) {
    if (typeof val !== "number" || !Number.isFinite(val) || val <= 0) return "";
    if (val >= 0.01) return "$" + val.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    return "$" + val.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }

  function renderUsagePillsHtml(usage) {
    if (!usage || typeof usage !== "object") return "";
    const pills = [];
    const orMeta = usage.openrouter_metadata && typeof usage.openrouter_metadata === "object"
      ? usage.openrouter_metadata
      : {};
    // Post-run reconciliation (GET /generation) adds native tokenizer counts and
    // upstream BYOK cost that inline metadata cannot report.
    const recon = usage.generation && typeof usage.generation === "object"
      ? usage.generation
      : {};

    let provider = typeof usage.provider === "string" ? usage.provider : "";
    if (!provider && Array.isArray(orMeta.endpoints)) {
      const sel = orMeta.endpoints.find(ep => ep && ep.selected && (ep.provider || ep.provider_name));
      if (sel) provider = String(sel.provider || sel.provider_name);
    }
    const attempt = typeof usage.attempt === "number" ? usage.attempt
      : (typeof orMeta.attempt === "number" ? orMeta.attempt : 1);
    const strategy = typeof usage.strategy === "string" ? usage.strategy
      : (typeof orMeta.strategy === "string" ? orMeta.strategy : "");
    const region = typeof usage.region === "string" ? usage.region
      : (typeof orMeta.region === "string" ? orMeta.region : "");
    if (provider) {
      const attemptSuffix = attempt > 1 ? ` · attempt ${attempt}` : "";
      const tooltipParts = [`Provider: ${provider}`];
      if (strategy) tooltipParts.push(`strategy: ${strategy}`);
      if (region) tooltipParts.push(`region: ${region}`);
      if (attempt > 1) tooltipParts.push(`attempt: ${attempt}`);
      pills.push(
        `<span class="meta-pill provider" title="${escapeHtml(tooltipParts.join(" · "))}">via ${escapeHtml(provider + attemptSuffix)}</span>`
      );
    } else if (attempt > 1) {
      pills.push(`<span class="meta-pill provider">attempt ${attempt}</span>`);
    }

    const genMs = typeof usage.generation_time === "number" ? usage.generation_time
      : (typeof orMeta.generation_time === "number" ? orMeta.generation_time : null);
    const elapsedMs = typeof usage.elapsed_ms === "number" ? usage.elapsed_ms : null;
    const displayMs = genMs !== null && genMs > 0 ? genMs : elapsedMs;
    if (typeof displayMs === "number" && displayMs >= 0) {
      const secText = displayMs >= 1000 ? (displayMs / 1000).toFixed(1) + "s" : Math.round(displayMs) + "ms";
      const timingTitle = genMs !== null && elapsedMs !== null
        ? `Generation: ${Math.round(genMs)}ms · Wall-clock: ${Math.round(elapsedMs)}ms`
        : (genMs !== null ? `Generation: ${Math.round(genMs)}ms` : `Latency: ${Math.round(displayMs)}ms`);
      pills.push(`<span class="meta-pill" title="${escapeHtml(timingTitle)}">⏱ ${escapeHtml(secText)}</span>`);
    }

    const totalTok = typeof usage.total_tokens === "number" ? usage.total_tokens : null;
    const promptTok = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : null;
    const compTok = typeof usage.completion_tokens === "number" ? usage.completion_tokens : null;
    const compDetails = usage.completion_tokens_details && typeof usage.completion_tokens_details === "object"
      ? usage.completion_tokens_details
      : {};
    const promptDetails = usage.prompt_tokens_details && typeof usage.prompt_tokens_details === "object"
      ? usage.prompt_tokens_details
      : {};
    const reasoningTok = typeof compDetails.reasoning_tokens === "number" && compDetails.reasoning_tokens > 0
      ? compDetails.reasoning_tokens
      : 0;
    const cachedTok = typeof promptDetails.cached_tokens === "number" && promptDetails.cached_tokens > 0
      ? promptDetails.cached_tokens
      : 0;

    if (totalTok !== null && totalTok > 0) {
      const details = [];
      if (reasoningTok > 0 && compTok !== null) {
        const visibleTok = Math.max(compTok - reasoningTok, 0);
        details.push(`${visibleTok} ans + ${reasoningTok} reasoning`);
      } else if (reasoningTok > 0) {
        details.push(`${reasoningTok} reasoning`);
      }
      if (cachedTok > 0) {
        details.push(`${cachedTok} cached`);
      }
      const label = details.length > 0
        ? `${totalTok} tok (${details.join(" · ")})`
        : `${totalTok} tok`;
      const tokTooltip = [];
      if (promptTok !== null) tokTooltip.push(`Prompt: ${promptTok}`);
      if (compTok !== null) tokTooltip.push(`Completion: ${compTok}`);
      if (reasoningTok > 0) tokTooltip.push(`Reasoning: ${reasoningTok}`);
      if (cachedTok > 0) tokTooltip.push(`Cached read: ${cachedTok}`);
      if (typeof recon.native_tokens_prompt === "number") {
        tokTooltip.push(`Native prompt: ${recon.native_tokens_prompt}`);
      }
      if (typeof recon.native_tokens_completion === "number") {
        tokTooltip.push(`Native completion: ${recon.native_tokens_completion}`);
      }
      pills.push(
        `<span class="meta-pill" title="${escapeHtml(tokTooltip.join(" · ") || label)}">${escapeHtml(label)}</span>`
      );
    }

    const toolUse = usage.server_tool_use && typeof usage.server_tool_use === "object"
      ? usage.server_tool_use
      : {};
    const searchReqs = typeof toolUse.web_search_requests === "number" && toolUse.web_search_requests > 0
      ? toolUse.web_search_requests
      : 0;
    const costDetails = usage.cost_details && typeof usage.cost_details === "object"
      ? usage.cost_details
      : {};
    const toolCost = typeof costDetails.server_tool_cost === "number" && costDetails.server_tool_cost > 0
      ? costDetails.server_tool_cost
      : 0;
    const totalCost = typeof usage.cost === "number" && usage.cost > 0 ? usage.cost : 0;

    if (searchReqs > 0) {
      const searchLabel = `${searchReqs} search${searchReqs === 1 ? "" : "es"}` + (toolCost > 0 ? ` (${formatUsd(toolCost)})` : "");
      pills.push(`<span class="meta-pill search" title="OpenRouter server-side web search">🌐 ${escapeHtml(searchLabel)}</span>`);
    }
    const cacheDiscount = typeof recon.cache_discount === "number" && recon.cache_discount > 0
      ? recon.cache_discount
      : 0;
    if (totalCost > 0) {
      const costSuffix = toolCost > 0 ? ` (includes ${formatUsd(toolCost)} web search)` : "";
      const cacheSuffix = cacheDiscount > 0 ? ` · cache discount ${formatUsd(cacheDiscount)}` : "";
      const costTitle = `Total cost: ${formatUsd(totalCost)}${costSuffix}${cacheSuffix}`;
      pills.push(`<span class="meta-pill cost" title="${escapeHtml(costTitle)}">${escapeHtml(formatUsd(totalCost))}</span>`);
    } else if (toolCost > 0 && searchReqs === 0) {
      pills.push(`<span class="meta-pill cost" title="Web search cost">${escapeHtml(formatUsd(toolCost))}</span>`);
    }

    const upstreamCost = typeof recon.upstream_inference_cost === "number" && recon.upstream_inference_cost > 0
      ? recon.upstream_inference_cost
      : 0;
    if (upstreamCost > 0) {
      const upstreamTitle = `Upstream provider cost (BYOK): ${formatUsd(upstreamCost)} · reconciled from GET /generation`;
      pills.push(`<span class="meta-pill cost" title="${escapeHtml(upstreamTitle)}">upstream ${escapeHtml(formatUsd(upstreamCost))}</span>`);
    }

    const genId = typeof usage.generation_id === "string" && usage.generation_id ? usage.generation_id : "";
    const nativeStop = typeof usage.native_finish_reason === "string" && usage.native_finish_reason
      ? usage.native_finish_reason
      : "";
    if (genId) {
      const shortGen = genId.length > 16 ? genId.slice(0, 14) + "…" : genId;
      const genTitle = nativeStop ? `Generation ID: ${genId} · stop: ${nativeStop}` : `Generation ID: ${genId}`;
      pills.push(`<span class="meta-pill" title="${escapeHtml(genTitle)}">${escapeHtml(shortGen)}</span>`);
    }

    return pills.join("");
  }

  function renderCitationsHtml(usage) {
    const citations = extractUsageCitations(usage);
    if (citations.length === 0) return "";
    const items = citations.map(c => {
      const snippet = c.content ? ` — <span>${escapeHtml(c.content.slice(0, 140))}${c.content.length > 140 ? "…" : ""}</span>` : "";
      return `<li><a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(c.title)}</a>${snippet}</li>`;
    }).join("");
    return `
      <details class="citations-strip">
        <summary>🔗 ${citations.length} web citation${citations.length === 1 ? "" : "s"}</summary>
        <ol class="citations-list">${items}</ol>
      </details>
    `;
  }

  function renderAnswerCard(ans, index, total) {
    const isError = Boolean(ans.error);
    const modelName = ans.model || ans.requested_model || "unknown-model";
    const routedNote = ans.requested_model && ans.model && ans.requested_model !== ans.model
      ? ` (via ${ans.requested_model})`
      : "";
    const pillsHtml = renderUsagePillsHtml(ans.usage);
    const citationsHtml = !isError ? renderCitationsHtml(ans.usage) : "";
    const cardClass = isError ? "answer-card error-card" : "answer-card";
    const statusBadge = isError
      ? '<span class="badge" style="color:var(--red);border-color:rgba(248,81,73,0.4);">failed</span>'
      : '<span class="badge">member ' + (index + 1) + ' of ' + total + '</span>';
    const bodyHtml = isError
      ? `<p style="color:#ff948f;font-family:var(--mono);font-size:13px;">${escapeHtml(ans.error)}</p>`
      : renderLatexDocument(ans.content || "");

    return `
      <article class="${cardClass}" data-answer-index="${index}">
        <header class="answer-header">
          <div class="answer-model">
            <span>🤖 ${escapeHtml(modelName + routedNote)}</span>
            ${statusBadge}
          </div>
          <div class="answer-meta">
            ${pillsHtml}
            ${!isError && ans.content ? `<button type="button" class="copy-btn" data-copy-text="${escapeHtml(ans.content)}">Copy</button>` : ""}
          </div>
        </header>
        <div class="markdown-body">${bodyHtml}</div>
        ${citationsHtml}
      </article>
    `;
  }

  function renderPromptCard(run) {
    const raw = run.prompt || "";
    return `
      <article class="answer-card">
        <header class="answer-header">
          <div class="answer-model">
            <span>📄 Stored prompt</span>
            <span class="badge">${raw.length} characters</span>
          </div>
          <div class="answer-meta">
            <button type="button" class="copy-btn" data-copy-text="${escapeHtml(raw)}">Copy</button>
          </div>
        </header>
        <div class="markdown-body">${renderLatexDocument(raw)}</div>
        <details class="raw-source">
          <summary>Raw prompt source</summary>
          <pre>${escapeHtml(raw)}</pre>
        </details>
      </article>
    `;
  }

  function renderSynthesisCard(run) {
    const judgeLabel = run.judge || run.judge_requested || "judge";
    const fallbackNote = run.judge_requested && run.judge && run.judge !== run.judge_requested
      ? ` (fallback from ${run.judge_requested})`
      : "";
    const elapsed = run.elapsed ? ` · ${run.elapsed}` : "";
    const synthUsage = run.synthesis_usage || {};
    const pillsHtml = renderUsagePillsHtml(synthUsage);
    const citationsHtml = renderCitationsHtml(synthUsage);
    return `
      <article class="answer-card synthesis-card" id="synthesis-card">
        <header class="answer-header">
          <div class="answer-model">
            <span>⚖️ Council Synthesis — ${escapeHtml(judgeLabel + fallbackNote)}</span>
            <span class="badge synth">Final Synthesis</span>
          </div>
          <div class="answer-meta">
            <span>${(run.answers || []).filter(a => a.content).length} valid answers${escapeHtml(elapsed)}</span>
            ${pillsHtml}
            <button type="button" class="copy-btn" data-copy-text="${escapeHtml(run.synthesis || "")}">Copy</button>
          </div>
        </header>
        <div class="markdown-body">${renderLatexDocument(run.synthesis || "")}</div>
        ${citationsHtml}
      </article>
    `;
  }

  function renderWorkspace() {
    const run = findRunById(state.selectedRunId);
    if (!run) {
      statusBanner.style.display = "none";
      paginationBar.style.display = "none";
      answersContainer.innerHTML = `
        <div class="empty-workspace" id="empty-workspace">
          <h2>Ready to convene the council</h2>
          <p>
            Enter a prompt above and click <strong>New</strong> to query all free OpenRouter models in parallel,
            or select a conversation from the left-hand history panel to inspect its answers or send a <strong>Followup</strong>.
          </p>
        </div>
      `;
      return;
    }

    const answers = run.answers || [];
    const hasSynthesis = Boolean(run.synthesis);
    const totalExpected = run.totalModels || answers.length;

    // Status Banner
    if (run.error) {
      statusBanner.className = "status-banner error";
      statusBanner.style.display = "flex";
      statusBanner.innerHTML = `<span>⚠️ ${escapeHtml(run.error)}</span>`;
    } else if (run.running) {
      statusBanner.className = "status-banner running";
      statusBanner.style.display = "flex";
      const stageText = run.stageText || `Receiving council answers (${answers.length}/${totalExpected || "?"})…`;
      statusBanner.innerHTML = `
        <span>⏳ ${escapeHtml(stageText)}</span>
        <span>Judge: ${escapeHtml(run.judge_requested || run.judge || getSelectedJudgeValue())}</span>
      `;
    } else if (run.source === "db" && !hasSynthesis && answers.length === 0) {
      statusBanner.className = "status-banner warning";
      statusBanner.style.display = "flex";
      statusBanner.innerHTML = `
        <span>ℹ️ Only the prompt was persisted in DuckDB for this run (initial answers and synthesis were not saved).</span>
      `;
    } else if (run.source === "db" && (!run.prompt || answers.length === 0 || !hasSynthesis)) {
      const missingParts = [];
      if (!run.prompt) missingParts.push("prompt text");
      if (answers.length === 0) missingParts.push("initial answers");
      if (!hasSynthesis) missingParts.push("synthesis");
      statusBanner.className = "status-banner";
      statusBanner.style.display = "flex";
      statusBanner.innerHTML = `
        <span>Loaded from DuckDB (subject to save settings — ${escapeHtml(missingParts.join(", "))} not stored for this run).</span>
      `;
    } else {
      statusBanner.style.display = "none";
    }

    // Pagination bar appears once synthesis arrives (or when browsing saved runs with answers)
    if (hasSynthesis || (!run.running && answers.length > 0)) {
      paginationBar.style.display = "flex";
      const pages = [];
      if (run.prompt) {
        pages.push({ key: "prompt", label: "📄 Prompt", cls: "" });
      }
      if (hasSynthesis) {
        pages.push({ key: "synthesis", label: "⚖️ Synthesis", cls: "synthesis-tab" });
      }
      answers.forEach((ans, idx) => {
        const shortName = (ans.model || ans.requested_model || `Model ${idx + 1}`).split("/").pop();
        const failedCls = ans.error ? "failed-tab" : "";
        pages.push({
          key: idx,
          label: `${idx + 1}. ${shortName}`,
          cls: failedCls,
        });
      });
      if (answers.length > 0) {
        pages.push({ key: "all", label: `All Answers (${answers.length})`, cls: "" });
      }

      const pageOrder = pages.map(p => p.key);
      if (!pageOrder.includes(state.activePage)) {
        state.activePage = hasSynthesis ? "synthesis" : (answers.length > 0 ? 0 : "all");
      }
      const curIdx = pageOrder.indexOf(state.activePage);

      const tabsHtml = pages.map(p => {
        const isActive = p.key === state.activePage ? "active" : "";
        return `<button type="button" class="page-tab ${p.cls} ${isActive}" data-page-key="${p.key}">${escapeHtml(p.label)}</button>`;
      }).join("");

      paginationBar.innerHTML = `
        <div class="page-tabs" role="tablist" aria-label="Answer pagination">
          ${tabsHtml}
        </div>
        <div class="pager-nav">
          <button type="button" class="pager-btn" id="prev-page-btn" ${curIdx <= 0 ? "disabled" : ""}>← Prev</button>
          <span style="font-family:var(--mono);font-size:11.5px;color:var(--text-muted);">
            Page ${curIdx + 1} of ${pageOrder.length}
          </span>
          <button type="button" class="pager-btn" id="next-page-btn" ${curIdx < 0 || curIdx >= pageOrder.length - 1 ? "disabled" : ""}>Next →</button>
        </div>
      `;

      paginationBar.querySelectorAll(".page-tab[data-page-key]").forEach(btn => {
        btn.addEventListener("click", function() {
          const rawKey = btn.getAttribute("data-page-key");
          const namedPage = rawKey === "synthesis" || rawKey === "all" || rawKey === "prompt";
          state.activePage = namedPage ? rawKey : Number(rawKey);
          renderWorkspace();
        });
      });

      const prevBtn = document.getElementById("prev-page-btn");
      const nextBtn = document.getElementById("next-page-btn");
      if (prevBtn) {
        prevBtn.addEventListener("click", function() {
          if (curIdx > 0) {
            state.activePage = pageOrder[curIdx - 1];
            renderWorkspace();
          }
        });
      }
      if (nextBtn) {
        nextBtn.addEventListener("click", function() {
          if (curIdx >= 0 && curIdx < pageOrder.length - 1) {
            state.activePage = pageOrder[curIdx + 1];
            renderWorkspace();
          }
        });
      }
    } else {
      paginationBar.style.display = "none";
    }

    // Main content rendering:
    // While streaming before synthesis arrives, show each answer immediately as it is generated.
    // Once synthesis appears, paginate between Synthesis, individual answers, or All Answers.
    if (!hasSynthesis && run.running) {
      if (answers.length === 0) {
        answersContainer.innerHTML = `
          <div class="empty-workspace">
            <h2>Querying council members…</h2>
            <p>Each model's answer will appear here automatically as soon as it finishes generating.</p>
          </div>
        `;
      } else {
        answersContainer.innerHTML = answers
          .map((ans, idx) => renderAnswerCard(ans, idx, totalExpected || answers.length))
          .join("");
      }
    } else if (state.activePage === "prompt" && run.prompt) {
      answersContainer.innerHTML = renderPromptCard(run);
    } else if (state.activePage === "synthesis" && hasSynthesis) {
      answersContainer.innerHTML = renderSynthesisCard(run);
    } else if (typeof state.activePage === "number" && answers[state.activePage]) {
      answersContainer.innerHTML = renderAnswerCard(
        answers[state.activePage],
        state.activePage,
        answers.length
      );
    } else if (state.activePage === "all") {
      const blocks = [];
      if (hasSynthesis) blocks.push(renderSynthesisCard(run));
      answers.forEach((ans, idx) => blocks.push(renderAnswerCard(ans, idx, answers.length)));
      answersContainer.innerHTML = blocks.join("") || '<div class="empty-workspace"><p>No answers recorded for this run.</p></div>';
    } else if (hasSynthesis) {
      answersContainer.innerHTML = renderSynthesisCard(run);
    } else {
      answersContainer.innerHTML = '<div class="empty-workspace"><p>No stored answers or synthesis for this conversation.</p></div>';
    }

    answersContainer.querySelectorAll(".copy-btn[data-copy-text]").forEach(btn => {
      btn.addEventListener("click", function() {
        const text = btn.getAttribute("data-copy-text") || "";
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text);
        }
        const orig = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = orig; }, 1200);
      });
    });
  }

  function buildPriorTurnsChain(targetRun) {
    const chain = [];
    let cur = targetRun;
    const visited = new Set();
    while (cur && !visited.has(cur.id)) {
      visited.add(cur.id);
      if (cur.prompt && cur.synthesis) {
        chain.unshift([cur.prompt, cur.synthesis]);
      }
      cur = cur.parentId ? findRunById(cur.parentId) : null;
    }
    return chain;
  }

  async function startCouncilQuery(mode) {
    if (state.isRunning) return;
    const rawPrompt = promptInput.value.trim();
    if (!rawPrompt && !state.attachedFile) {
      promptInput.focus();
      return;
    }

    const parentRun = mode === "followup" ? findRunById(state.selectedRunId) : null;
    if (mode === "followup" && (!parentRun || !parentRun.synthesis)) {
      return;
    }

    const tempId = "session-" + Date.now();
    const displayPrompt = rawPrompt || (state.attachedFile ? `[Attached: ${state.attachedFile.name}]` : "");
    const judgeValue = getSelectedJudgeValue();

    const newRun = {
      id: tempId,
      created_at: Date.now() / 1000,
      prompt: displayPrompt,
      answers: [],
      synthesis: null,
      synthesis_usage: {},
      judge: null,
      judge_requested: judgeValue,
      running: true,
      stageText: "Discovering free council models and validating judge…",
      totalModels: 0,
      source: "session",
      isFollowup: mode === "followup",
      parentId: parentRun ? parentRun.id : null,
    };

    state.sessionRuns.unshift(newRun);
    state.selectedRunId = tempId;
    state.activePage = "all";
    const abortController = new AbortController();
    state.activeAbortController = abortController;
    state.isRunning = true;
    newBtn.disabled = true;
    stopBtn.disabled = false;
    updateFollowupButtonState();
    renderHistorySidebar();
    renderWorkspace();

    const payload = {
      prompt: rawPrompt,
      file_name: state.attachedFile ? state.attachedFile.name : null,
      file_content: state.attachedFile ? state.attachedFile.content : null,
      judge: judgeValue,
      search_web: searchWebCheckbox.checked,
      previous_synthesis: null,
      prior_turns: null,
    };

    if (mode === "followup" && parentRun) {
      const chain = buildPriorTurnsChain(parentRun);
      if (chain.length > 0) {
        payload.prior_turns = chain;
      } else if (parentRun.synthesis) {
        payload.previous_synthesis = parentRun.synthesis;
      }
    }

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });

      if (!response.ok || !response.body) {
        let errText = "Request failed (" + response.status + ")";
        try {
          const errJson = await response.json();
          if (errJson && errJson.error) errText = errJson.error;
        } catch (_) {}
        throw new Error(errText);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const chunk of parts) {
          const dataLine = chunk
            .split("\n")
            .find(line => line.startsWith("data:"));
          if (!dataLine) continue;
          const jsonStr = dataLine.slice(5).trim();
          if (!jsonStr) continue;
          let evt;
          try {
            evt = JSON.parse(jsonStr);
          } catch (_) {
            continue;
          }
          handleStreamEvent(newRun, evt);
        }
      }
    } catch (err) {
      if (err && err.name === "AbortError") {
        newRun.error = "Stopped by user.";
      } else {
        newRun.error = err && err.message ? err.message : String(err);
      }
      newRun.running = false;
    } finally {
      newRun.running = false;
      if (state.activeAbortController === abortController) {
        state.activeAbortController = null;
      }
      state.isRunning = false;
      newBtn.disabled = false;
      stopBtn.disabled = true;
      updateFollowupButtonState();
      renderHistorySidebar();
      renderWorkspace();
    }
  }

  function handleStreamEvent(run, evt) {
    if (evt.type === "discovered") {
      if (evt.question_id) {
        const wasSelected = state.selectedRunId === run.id;
        run.id = evt.question_id;
        if (wasSelected) state.selectedRunId = run.id;
      }
      if (evt.prompt) run.prompt = evt.prompt;
      run.totalModels = (evt.models || []).length;
      run.judge_requested = evt.judge || run.judge_requested;
      run.stageText = `Querying ${run.totalModels} council models in parallel…`;
    } else if (evt.type === "member_answer") {
      run.answers.push({
        requested_model: evt.requested_model,
        model: evt.model,
        content: evt.content,
        error: evt.error,
        usage: evt.usage || {},
      });
      run.stageText = `Received ${run.answers.length} of ${evt.total || run.totalModels} council answers…`;
    } else if (evt.type === "synthesizing") {
      run.stageText = `Synthesizing ${evt.valid_count} valid answers with judge ${evt.judge}…`;
    } else if (evt.type === "synthesis") {
      run.synthesis = evt.synthesis;
      run.synthesis_usage = evt.usage || {};
      run.judge = evt.judge;
      run.judge_requested = evt.judge_requested || run.judge_requested;
      run.elapsed = evt.elapsed;
      run.running = false;
      if (state.selectedRunId === run.id) {
        state.activePage = "synthesis";
      }
    } else if (evt.type === "error") {
      run.error = evt.error || "Council run failed.";
      run.running = false;
    }
    updateFollowupButtonState();
    renderHistorySidebar();
    if (state.selectedRunId === run.id) {
      renderWorkspace();
    }
  }

  async function requestPromptImprovements() {
    if (state.isRunning) return;
    const rawPrompt = promptInput.value.trim();
    if (!rawPrompt && !state.attachedFile) {
      promptInput.focus();
      return;
    }
    const abortController = new AbortController();
    state.activeAbortController = abortController;
    state.isRunning = true;
    improveBtn.disabled = true;
    newBtn.disabled = true;
    stopBtn.disabled = false;
    updateFollowupButtonState();
    improveDrawer.classList.add("open");
    improveHeaderTitle.textContent = "Generating improved prompt versions with a single LLM…";
    improveOptions.innerHTML = "";

    try {
      const resp = await fetch("/api/improve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: rawPrompt,
          file_name: state.attachedFile ? state.attachedFile.name : null,
          file_content: state.attachedFile ? state.attachedFile.content : null,
          judge: getSelectedJudgeValue(),
        }),
        signal: abortController.signal,
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        throw new Error(data.error || "Failed to improve prompt");
      }
      const prompts = data.prompts || [];
      improveHeaderTitle.textContent = `Suggested Prompt Improvements (${prompts.length}) — click any version to populate composer:`;
      improveOptions.innerHTML = prompts.map((p, idx) => `
        <button type="button" class="improve-option-btn" data-improved-idx="${idx}">
          <strong>${idx + 1}.</strong> ${escapeHtml(p)}
        </button>
      `).join("");

      improveOptions.querySelectorAll(".improve-option-btn").forEach((btn, idx) => {
        btn.addEventListener("click", function() {
          promptInput.value = prompts[idx];
          promptInput.focus();
        });
      });
    } catch (err) {
      if (err && err.name === "AbortError") {
        improveHeaderTitle.textContent = "Prompt improvement stopped by user.";
      } else {
        improveHeaderTitle.textContent = "Prompt improvement error: " + (err.message || String(err));
      }
    } finally {
      if (state.activeAbortController === abortController) {
        state.activeAbortController = null;
      }
      state.isRunning = false;
      improveBtn.disabled = false;
      newBtn.disabled = false;
      stopBtn.disabled = true;
      updateFollowupButtonState();
    }
  }

  async function loadInitialState() {
    try {
      const [cfgResp, histResp] = await Promise.all([
        fetch("/api/config"),
        fetch("/api/history"),
      ]);
      if (cfgResp.ok) {
        const cfg = await cfgResp.json();
        if (typeof cfg.search_web === "boolean") {
          searchWebCheckbox.checked = cfg.search_web;
        }
        if (cfg.judge) {
          const builtIns = ["random", "round-robin", "most-intelligent", "largest-context", "free-router", "auto"];
          if (builtIns.includes(cfg.judge)) {
            judgeSelect.value = cfg.judge;
          } else {
            judgeSelect.value = "custom";
            judgeCustomInput.style.display = "inline-block";
            judgeCustomInput.value = cfg.judge;
          }
        }
      }
      if (histResp.ok) {
        const data = await histResp.json();
        state.dbRuns = Array.isArray(data.history) ? data.history : [];
      }
    } catch (_) {
      // Keep UI functional even if initial history fetch fails
    }
    renderHistorySidebar();
    updateFollowupButtonState();
  }

  // Event wiring
  searchWebCheckbox.addEventListener("change", function() {
    searchWebToggle.classList.toggle("enabled", searchWebCheckbox.checked);
  });

  judgeSelect.addEventListener("change", function() {
    if (judgeSelect.value === "custom") {
      judgeCustomInput.style.display = "inline-block";
      judgeCustomInput.focus();
    } else {
      judgeCustomInput.style.display = "none";
    }
  });

  attachFileBtn.addEventListener("click", function() {
    fileInput.click();
  });

  fileInput.addEventListener("change", function() {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(evt) {
      state.attachedFile = {
        name: file.name,
        content: String(evt.target && evt.target.result ? evt.target.result : ""),
      };
      renderAttachmentChip();
    };
    reader.readAsText(file);
  });

  clearBtn.addEventListener("click", function() {
    promptInput.value = "";
    promptInput.focus();
  });

  stopBtn.addEventListener("click", function() {
    if (state.activeAbortController) {
      state.activeAbortController.abort();
    }
  });

  newBtn.addEventListener("click", function() {
    startCouncilQuery("new");
  });

  followupBtn.addEventListener("click", function() {
    startCouncilQuery("followup");
  });

  improveBtn.addEventListener("click", function() {
    requestPromptImprovements();
  });

  closeImproveBtn.addEventListener("click", function() {
    improveDrawer.classList.remove("open");
  });

  historyFilterInput.addEventListener("input", function() {
    state.filterText = historyFilterInput.value || "";
    renderHistorySidebar();
  });

  promptInput.addEventListener("keydown", function(evt) {
    if ((evt.ctrlKey || evt.metaKey) && evt.key === "Enter") {
      evt.preventDefault();
      if (evt.shiftKey && !followupBtn.disabled) {
        startCouncilQuery("followup");
      } else {
        startCouncilQuery("new");
      }
    }
  });

  loadInitialState();
})();
</script>
</body>
</html>
"""


def _combine_prompt_and_attachment(
    prompt: str | None,
    file_content: str | None,
) -> str:
    """Combine inline prompt text and optional attached file content."""
    clean_prompt = (prompt or "").strip()
    clean_file = (file_content or "").strip()
    if clean_prompt and clean_file:
        return f"{clean_prompt}\n\n{clean_file}"
    return clean_prompt or clean_file


def create_web_server(
    *,
    host: str = DEFAULT_WEB_HOST,
    port: int = DEFAULT_WEB_PORT,
    env_file: str | Path | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer serving the llm-council web interface and JSON/SSE API."""
    base_settings = (settings or Settings.from_env(env_file)).with_overrides(**overrides)

    class CouncilWebHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("Web HTTP %s - %s", self.address_string(), format % args)

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> dict[str, Any]:
            length_header = self.headers.get("Content-Length", "0")
            try:
                length = int(length_header)
            except ValueError as exc:
                raise CouncilError("Invalid Content-Length header.") from exc
            if length <= 0:
                return {}
            if length > MAX_REQUEST_BYTES:
                raise CouncilError("Request payload exceeds maximum allowed size.")
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CouncilError(f"Invalid JSON request body: {exc}") from exc
            if not isinstance(parsed, dict):
                raise CouncilError("JSON request body must be an object.")
            return parsed

        def _serve_static(self, url_path: str) -> None:
            """Serve a vendored asset from STATIC_DIR, refusing any path escape."""
            relative = unquote(url_path[len(STATIC_URL_PREFIX) :])
            candidate = (STATIC_DIR / relative).resolve()
            try:
                candidate.relative_to(STATIC_DIR)
            except ValueError:
                logger.warning("Rejected static path escaping STATIC_DIR: %s", url_path)
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            body = candidate.read_bytes()
            content_type = STATIC_CONTENT_TYPES.get(
                candidate.suffix.lower(), "application/octet-stream"
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            if parsed_url.path.startswith(STATIC_URL_PREFIX):
                self._serve_static(parsed_url.path)
                return

            if parsed_url.path in ("/", "/index.html"):
                body = _UI_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed_url.path == "/api/config":
                self._send_json(
                    {
                        "judge": base_settings.judge,
                        "strategies": list(STRATEGIES),
                        "save_prompt": base_settings.save_prompt.value,
                        "save_initial_answers": base_settings.save_initial_answers.value,
                        "save_synthesis": base_settings.save_synthesis.value,
                        "search_web": base_settings.search_web,
                        "time_zone": os.environ.get("TIME_ZONE"),
                    }
                )
                return

            if parsed_url.path == "/api/history":
                try:
                    history = load_history(settings=base_settings)
                    self._send_json({"history": history})
                except CouncilError as exc:
                    self._send_json(
                        {"error": str(exc), "history": []}, status=HTTPStatus.BAD_REQUEST
                    )
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed_url = urlparse(self.path)
            if parsed_url.path == "/api/improve":
                try:
                    body = self._read_json_body()
                    combined = _combine_prompt_and_attachment(
                        body.get("prompt"),
                        body.get("file_content"),
                    )
                    judge = body.get("judge") or base_settings.judge
                    prompts = improve_prompt(
                        combined,
                        settings=base_settings,
                        judge=str(judge),
                        literal_prompt=True,
                    )
                    self._send_json({"prompts": list(prompts)})
                except CouncilError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                except Exception as exc:
                    logger.error("Unexpected error in /api/improve: %s", exc)
                    self._send_json(
                        {"error": f"Unexpected server error: {exc}"},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return

            if parsed_url.path == "/api/query":
                try:
                    body = self._read_json_body()
                    combined = _combine_prompt_and_attachment(
                        body.get("prompt"),
                        body.get("file_content"),
                    )
                    if not combined:
                        self._send_json(
                            {"error": "Provide a non-empty prompt or attached file."},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return

                    judge = body.get("judge") or base_settings.judge
                    search_web = body.get("search_web")
                    if search_web is not None and not isinstance(search_web, bool):
                        raise CouncilError("search_web must be a JSON boolean.")
                    previous_synthesis = body.get("previous_synthesis")
                    raw_prior_turns = body.get("prior_turns")
                    prior_turns: list[tuple[str, str]] | None = None
                    if isinstance(raw_prior_turns, list):
                        prior_turns = [
                            (str(turn[0]), str(turn[1]))
                            for turn in raw_prior_turns
                            if isinstance(turn, (list, tuple)) and len(turn) == 2
                        ]
                except CouncilError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Connection", "close")
                self.end_headers()

                cancel_event = Event()

                def send_sse(event_payload: dict[str, Any]) -> None:
                    try:
                        line = f"data: {json.dumps(event_payload)}\n\n".encode()
                        self.wfile.write(line)
                        self.wfile.flush()
                    except OSError as write_exc:
                        cancel_event.set()
                        logger.debug("SSE client disconnected: %s", write_exc)

                try:
                    query_result(
                        combined,
                        settings=base_settings,
                        judge=str(judge),
                        search_web=search_web,
                        literal_prompt=True,
                        previous_synthesis=(
                            str(previous_synthesis)
                            if isinstance(previous_synthesis, str) and previous_synthesis.strip()
                            else None
                        ),
                        prior_turns=prior_turns,
                        on_event=send_sse,
                        cancel_event=cancel_event,
                    )
                except CouncilError as exc:
                    if not cancel_event.is_set():
                        logger.warning("Council web query error: %s", exc)
                        send_sse({"type": "error", "error": str(exc)})
                except Exception as exc:
                    if not cancel_event.is_set():
                        logger.error("Unexpected council web query failure: %s", exc)
                        send_sse({"type": "error", "error": f"Unexpected error: {exc}"})
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((host, port), CouncilWebHandler)


def serve_web(
    *,
    host: str = DEFAULT_WEB_HOST,
    port: int = DEFAULT_WEB_PORT,
    open_browser: bool = False,
    env_file: str | Path | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> None:
    """Run the interactive llm-council web UI server until interrupted."""
    server = create_web_server(
        host=host,
        port=port,
        env_file=env_file,
        settings=settings,
        **overrides,
    )
    raw_host, actual_port = server.server_address[:2]
    actual_host = raw_host.decode() if isinstance(raw_host, bytes) else str(raw_host)
    url = f"http://{actual_host}:{actual_port}"
    logger.info("Starting llm-council web UI at %s (started_at=%d)", url, int(time.time()))
    print(f"llm-council web UI listening on {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping llm-council web UI server")
    finally:
        server.server_close()
