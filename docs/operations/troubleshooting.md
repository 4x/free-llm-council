# Troubleshooting

Operational failure modes for `llm-council`. Logs go to stderr; synthesis goes to stdout.

## Quota and rate limits: 402 vs 429/503

- `429` / `503` with `Retry-After` are retried with backoff, up to `MAX_RETRIES`. Waits longer than `MAX_RETRY_WAIT` fail fast instead of sleeping indefinitely.
- `402` is retried only when `error.metadata.limit_source` is `openrouter_in_flight_budget`. Other `402` (insufficient credits) fails immediately — top up credits, don't raise retries.
- What to check: run with `--log-level debug` to see `Retry-After`, `limit_source`, and redacted provider metadata. Lower `MAX_CONCURRENCY` or raise `REQUEST_INTERVAL` (default `3` / `3.1s` targets <20 starts/min). Free quotas are account-wide.

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

## DuckDB locked / missing tables

- Do not run concurrent writer processes against the same DuckDB file; use separate files or serialize runs.
- `DuckDB file X does not exist` on export means `DUCKDB_PATH` is wrong or the run used `--no-save` / markdown-only modes.
- Responses-only runs create a `questions` identity row with NULL `prompt`/`source_path` — prompt text is never silently saved.

## Auth and config

- `Set OPENROUTER_API_KEY` means the key is missing after `.env` → env → override precedence. `Loading .env does not mutate process environment`.
- `JUDGE_PROMPT` unreadable/invalid-UTF-8/blank fails before discovery, inference, or saving. Unset uses the built-in backup.
