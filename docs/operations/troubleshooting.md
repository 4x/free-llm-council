# Troubleshooting

Operational failure modes for `llm-council`. Logs go to stderr; synthesis goes to stdout.

## Quota and rate limits: 402 vs 429/503

- `429` / `503` with `Retry-After` are retried with backoff, up to `MAX_RETRIES`. Waits longer than `MAX_RETRY_WAIT` fail fast instead of sleeping indefinitely.
- `402` is retried only when `error.metadata.limit_source` is `openrouter_in_flight_budget`. Other `402` (insufficient credits) fails immediately — top up credits, don't raise retries.
- What to check: run with `--log-level debug` to see `Retry-After`, `limit_source`, and redacted provider metadata. Lower `MAX_CONCURRENCY` or raise `REQUEST_INTERVAL` (default `3` / `3.1s` targets <20 starts/min). Free quotas are account-wide.

## Truncated answers: `exhausted MAX_TOKENS`

Symptom: a member fails with `exhausted MAX_TOKENS: ...`. The suffix identifies what consumed the budget.

- `reasoning used all N completion tokens` — the model spent the whole budget thinking. Raising `MAX_TOKENS` will not produce an answer; cap reasoning with `reasoning.max_tokens`/`reasoning.effort`, or drop the model (see [OpenRouter API capabilities](../reference/openrouter-api-capabilities.md)).
- `reasoning used N of M completion tokens` — reasoning-dominated. Raising `MAX_TOKENS` mostly buys more reasoning, so cap it instead.
- `the answer used N of M completion tokens` — genuine long answer. Increase `MAX_TOKENS`.
- `no reasoning-token accounting` — the provider returned no `usage.completion_tokens_details`; the cause is unknown, so try raising `MAX_TOKENS` or capping reasoning.

Raw per-answer usage (including `completion_tokens_details`) is persisted in DuckDB, so a truncated run can be inspected after the fact. Truncation is still a failure even when partial text was returned; the message notes when a partial answer was discarded.

## Harness-only / restricted models: 403

Symptom: one model always fails with `403`, e.g. `thinkingmachines/inkling-small:free` returning harness-only.

Fix: add it to `EXCLUDED_MODELS`:

```bash
uv run llm-council -q "test" --excluded-models thinkingmachines/inkling-small:free --no-save
```

```env
EXCLUDED_MODELS=thinkingmachines/inkling-small:free
```

Exclusions apply to members and judges, including routers and paid judges. There is no built-in blacklist; `DEBUG` logs show skipped models.

## Web search problems

`--search-web` / `SEARCH_WEB=1` adds OpenRouter's `openrouter:web_search` server tool (capped at 3 searches per request) to every member and judge call; `GEMINI_MODELS` use Google Search grounding instead.

- **Unexpected cost.** Search fees are separate from token costs and are not covered by the free-model price cap, so free models still incur them. Free models have no native search, so `auto` runs Exa at $0.007 per search and our `max_uses: 3` caps a single request at $0.021; a model that skips searching adds nothing. `DEBUG` logs `Model X ran N web search(es)` from `usage.server_tool_use.web_search_requests`; turn `SEARCH_WEB` off or remove the models that search most. `usage.cost` in the raw response is the authoritative total actually debited, including search fees.
- **Every member fails with `None of the N free text models supports the tool calling that --search-web requires`.** Discovery kept only models whose `supported_parameters` includes `tools`, and none qualified. Disable `SEARCH_WEB` or add Gemini models, which search through the Gemini API without needing OpenRouter tool support.
- **`ConfigurationError` naming the judge.** An explicit judge that cannot call tools is rejected before member requests when search is on; choose a tool-capable judge, use an automatic strategy (which filters to tool-capable models), or disable `SEARCH_WEB`. An explicit model not found in the catalog at all is reported as an unknown judge instead.
- **Answers arrive without fresh information.** The server tool is model-decided: a model may skip searching when it believes it already knows, and `engine: "auto"` prefers provider-native search before falling back to Exa. Check `web_search_requests` at `DEBUG` to see whether the model searched at all.
- **Gemini grounding fails.** Google Search grounding is only available for supported Gemini models and can be region- or quota-restricted. The run reports the Gemini error unchanged; drop the model, or use OpenRouter-hosted models for that run.

## Judge fallback

Normal behavior, not an error:

- `Judge X failed: ...` in logs, then `Synthesizing N valid answers with Y` means the selected judge failed and the run fell back to a successful council member (excluding any model that already errored this run).
- If the judge was also a failed member, it is never retried for synthesis.
- Inspect `result.judge` vs `result.judge_requested` (Python) — they differ after fallback.
- If judge + all fallbacks fail, the run raises `Judge and successful-model fallbacks failed`. Initial answers are still in DuckDB when that stage is enabled. Use `largest-context` or an explicit large-context judge for big councils.

## No Markdown created

Expected when synthesis itself fails:

- Every member fails, or judge + fallbacks fail → no folder or file is created. Check stderr for `No valid answers` or `fallbacks failed`.
- Otherwise check save modes: defaults are `SAVE_PROMPT=duckdb`, `SAVE_INITIAL_ANSWERS=none`, `SAVE_SYNTHESIS=none`. Synthesis-only Markdown needs:

```env
SAVE_PROMPT=none
SAVE_INITIAL_ANSWERS=none
SAVE_SYNTHESIS=markdown
```

That produces a single file directly under `council-output/`:

```text
council-output/<slug>-synthesis-<YYYYMMDDThhmm>.md
```

Multi-file runs produce a folder instead. Empty responses are never written. `--no-save` suppresses everything, including `--export-markdown` combinations.

## Export skips

`--export-markdown` / `export_markdown()` skips runs whose target folder/file and contents already exist. Re-running is safe and idempotent. It only exports runs with a non-empty synthesis and non-empty `expected_files` for the active save modes.

`--export-html` / `export_html()` likewise exports only runs with a non-empty synthesis, and skips a run when a file matching its name already holds byte-identical content. Changing report rendering therefore produces a new suffixed file rather than overwriting an existing one; delete the old report if you want the name reused.

## HTML report problems

- **The report opens but shows raw Markdown and LaTeX.** The report typesets in the viewer, so it needs JavaScript. The `<noscript>` block shows the raw synthesis meanwhile. A CSP that blocks inline scripts has the same effect.
- **Math shows as a red error.** KaTeX rejected the construct and it was declared inside `$...$`, `$$...$$`, or `\[...\]`, which is taken at face value and shown as authored. Bare environments behave differently — see [LaTeX document problems](#latex-document-problems).
- **`Expected 20 KaTeX @font-face rules … but inlined N`.** The vendored `static/katex/katex.min.css` no longer matches the shape the inliner expects (see `report.py`). Re-vendor consistently or update `_FONT_FACE_RE`; the export deliberately fails instead of writing a report that points at fonts it cannot embed.
- **`Cannot read bundled asset …` / `Cannot read bundled font …`.** The package's `static/` assets are missing — usually a source checkout that lost them, or an install that did not include package data. Reinstall the package.
- **Reports are ~650 KB each.** Expected: 20 `woff2` fonts are embedded so the file stands alone. The size is the price of working with no network.

## LaTeX document problems

Stored prompts are often pasted LaTeX documents; [docs/architecture/latex-documents.md](../architecture/latex-documents.md) records the rules these symptoms come from.

- **A block shows its literal source in a highlighted box.** KaTeX rejected that block, so it is printed rather than rendered as wrong math. Usual causes: a command KaTeX does not define (`\label`, `\ref`, `\caption`, `\centering`, `\bigskip`) or a nested environment the extractor cannot follow. The surrounding document still renders.
- **A line starting with `%` vanished.** It is a LaTeX comment. Comments are omitted from the rendered view because commented-out math would otherwise be typeset as if it were part of the document. Open **Raw prompt source** to see it.
- **Prose looks truncated after a percent sign.** It should not: a `%` is only a comment at the start of a line or directly after `}`, `\`, `$`, or `]`. A `%` after a word or digit is kept, which is why `~20% of samples` survives.
- **Something like `[semi]` appears where a number should be.** That is a `\ref{semi}` with nothing numbered to point at. The common case is a synthesis or answer referring to a label defined in the prompt: each field is numbered on its own, so printing the prompt's number in the synthesis would disagree with the synthesis's own numbering, and the target is shown instead. Check the **Prompt** page or the report's Prompt section for the number. If the label is defined in the same field and still shows bracketed, its equation was not numbered — which happens only when the block carrying it has the author's own `\tag`/`\notag`/`\nonumber`, is an `equation*`, or is a row-based environment such as `eqnarray`/`align`/`gather` rather than a plain `equation`.
- **An equation shows no number beside it.** Only a plain `equation` block is numbered. `equation*`, `eqnarray`, `align`, and `gather` blocks are rendered without a number, as are equations whose author wrote `\tag` or `\notag`. KaTeX itself numbers nothing — an empty `katex-tag` element is all it produces — so the number comes from the renderer, and a block it does not number simply has none.
- **Stray `$`, `<`, or `<y<1$` in the text.** The stored text itself has damaged `$` delimiters — common in material copied out of a PDF, where parts of formulas were lost before the prompt was ever saved. Delimiters that are not paired are left as literal text; they are not repaired, because deciding whether a lone `$` is math or currency is not reliably decidable.
- **`eqnarray` still renders.** It is mapped onto KaTeX's `aligned`, so a document written with the older name typesets; its `&=&` bodies are retried as `aligned` when the original environment rejects them.

## Reconciliation failures (`--reconcile-generations`)

Symptom: the pass prints rows under the summary and exits with status 1.

- `OpenRouter error 404: ...` is handled, not a failure: the generation was pruned, so the row is tagged `generation_unavailable` and never fetched again. Nothing else to do.
- Any other error (401/403, 429, 5xx, transport) leaves the row queued and is retried on the next run. Fix the cause — usually `OPENROUTER_API_KEY` or a temporary outage — then re-run; the summary's failure lines name the model and the full provider message.
- `No stored answers are awaiting reconciliation; N have no generation ID` means the stored rows predate metadata capture or came from Gemini. There is nothing to fetch; native counts for those runs cannot be recovered.
- `DuckDB file ... does not exist` means the run history lives elsewhere; point `--duckdb-path`/`DUCKDB_PATH` at the right file.

## DuckDB locked / missing tables

- Do not run concurrent writer processes against the same DuckDB file; use separate files or serialize runs.
- `DuckDB file X does not exist` on export means `DUCKDB_PATH` is wrong or the run used `--no-save` / markdown-only modes.
- Responses-only runs create a `questions` identity row with NULL `prompt`/`source_path` — prompt text is never silently saved.

## Auth and config

- `Set OPENROUTER_API_KEY` means the key is missing after `.env` → env → override precedence. `Loading .env does not mutate process environment`.
- `Set GEMINI_API_KEY in .env or the environment when GEMINI_MODELS is configured` means `GEMINI_MODELS` (or `--gemini-models`) is non-empty without a corresponding `GEMINI_API_KEY`.
- `JUDGE_PROMPT` unreadable/invalid-UTF-8/blank fails before discovery, inference, or saving. Unset uses the built-in backup.
