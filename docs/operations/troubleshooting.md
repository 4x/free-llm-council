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

## DuckDB locked / missing tables

- Do not run concurrent writer processes against the same DuckDB file; use separate files or serialize runs.
- `DuckDB file X does not exist` on export means `DUCKDB_PATH` is wrong or the run used `--no-save` / markdown-only modes.
- Responses-only runs create a `questions` identity row with NULL `prompt`/`source_path` — prompt text is never silently saved.

## Auth and config

- `Set OPENROUTER_API_KEY` means the key is missing after `.env` → env → override precedence. `Loading .env does not mutate process environment`.
- `Set GEMINI_API_KEY in .env or the environment when GEMINI_MODELS is configured` means `GEMINI_MODELS` (or `--gemini-models`) is non-empty without a corresponding `GEMINI_API_KEY`.
- `JUDGE_PROMPT` unreadable/invalid-UTF-8/blank fails before discovery, inference, or saving. Unset uses the built-in backup.
