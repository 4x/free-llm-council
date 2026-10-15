# llm-council

Ask **every currently free OpenRouter text model**, then ask a judge LLM to critically
synthesize all successful answers. Provides a Python API and `llm-council` CLI.
Requires Python 3.11+ and an OpenRouter API key.

## Install

```bash
uv sync
cp .env.example .env
# Set OPENROUTER_API_KEY in .env before querying.
uv run llm-council --help
```

For editable installation into another application's environment:

```bash
pip install -e /path/to/llm-council
# Or: uv pip install -e /path/to/llm-council
```

The distribution/CLI name is `llm-council`; Python imports use `llm_council`
(hyphens are not legal in Python module names). No API key is included.

## Python

```python
from llm_council import improve_prompt, query, query_result

# Ask a single LLM to rewrite a prompt into several improved versions:
candidates = improve_prompt("Explain how beta distribution is related to Gaussian")

synthesized_response1 = query(
    prompt=candidates[0],
    save_initial_answers="both",
    judge="random",
)

synthesized_response2 = query(
    prompt="path/to/long-prompt.txt",  # existing file paths are read automatically
    save_synthesis="neither",
    judge="most-intelligent",
)

# Explicit file input is unambiguous; UTF-8 .txt/.md (or other text) files work.
answer = query(file="path/to/prompt.md", no_save=True)

# Structured result for diagnostics and actual model identities:
result = query_result(
    prompt="Explain the Riemann hypothesis",
    judge="largest-context",
    save_prompt="both",
    save_initial_answers="both",
    save_synthesis="both",
)
print(result.synthesis)
print(result.question_id, result.judge, result.markdown_path)
for failure in result.failures:
    print(failure.requested_model, failure.error)
```

`query()` returns `str`. `improve_prompt()` calls a single LLM (selected via the configured
`judge` policy, falling back to other free models on error) and returns `tuple[str, ...]`
of improved prompt versions. `query_result()` returns `CouncilResult` with the original
prompt, initial `ModelAnswer` records (including failures), synthesized text, requested
and actual judge IDs, usage, question ID, and optional Markdown folder or single synthesis file. `ModelAnswer`
contains `requested_model`, actual `model`, `content`, `error`, and `usage`.

Both functions accept `Path` input, `env_file`, a `Settings` object, and overrides for
any setting below using its lowercase Python name. Set `literal_prompt=True` to treat
a string as text even if it names an existing file; CLI `--question` always uses literal
text. Invalid input/configuration and provider/storage failures raise a `CouncilError`
subtype. Member failures emit a logging warning but do not discard successful answers
or abort when Python warnings are configured as errors. Synthesis proceeds with **at least
one valid, non-empty answer**; failed, empty, and token-truncated answers are excluded.
If every member fails, no judge is called.

## CLI

```bash
uv run llm-council --question "Explain the Riemann hypothesis" --save-initial-answers markdown
uv run llm-council --improve "Explain the Riemann hypothesis"
uv run llm-council --prompt 2
uv run llm-council --file path/to/prompt.md --save-synthesis duckdb
uv run llm-council -q "Compare Bayesian and frequentist inference" --judge largest-context \
  --save-prompt both --save-initial-answers both --save-synthesis both
uv run llm-council -q "Explain entropy" --no-save
uv run llm-council --followup "provide more details about that last security caveat" --copy
uv run llm-council --export-markdown
uv run llm-council --web --port 8765
uv run llm-council -q "Explain entropy" --judge auto --judge-scope all
python -m llm_council --help
```

Supply `--question`, `--file`, or both, or use `--improve` / `--improve-prompt` (to ask one LLM
for numbered improved versions of a prompt, cached in `.llm-council/improved-prompts.json` unless
`--no-save` is passed) followed by `--prompt <n>` (1-based) to run the council with the `n`th
improved prompt, or pass `--web` / `--serve` (optionally with `--port`, default `8765`) to launch
the local HTML web interface.

### Web UI (`--web` / `--serve`)

`uv run llm-council --web` starts a local HTTP/SSE workspace on `http://127.0.0.1:8765` with:
- A prominent prompt composer with `<New>`, `<Followup>`, `<Clear>` (clears the prompt textbox), and red `<Stop>` (disabled/grayed out unless an active LLM retrieval is in progress) buttons with hover tooltips, a paperclip file attachment button (`📎 Attach file`), a Judge Criterion selector (all built-in strategies plus custom model slug), and a one-click `✨ Improve Prompt` drawer.
- Real-time Server-Sent Events (SSE) streaming of council member answers as each model finishes, followed by the synthesized response and pagination pills (`Synthesis`, numbered member tabs, `Prev`/`Next`, and `All Answers`).
- KaTeX typesetting of LaTeX in answers and syntheses. Inline `$...$` and `\(...\)` render inline; `$$...$$` and `\[...\]` render as centered display equations that scroll horizontally when they are wider than the card. Math inside fenced or inline code stays literal, as do currency amounts like `$5` and escaped `\$`.
- Left-hand conversation history panel separating **Current Session** runs from **Saved in DuckDB** runs (rendered less prominently). Selecting any historical run populates its prompt and stored answers/synthesis (subject to availability), and clicking `<Followup>` while a conversation is selected follows up directly on that conversation.

The KaTeX bundle that renders the LaTeX is **vendored, not fetched from a CDN**, so saved
conversations keep rendering correctly with no network access. The assets live in
`src/llm_council/static/katex/` and are served by the web server from `/static/` (it refuses
any path that escapes `STATIC_DIR`). Only the `woff2` font format is vendored, because
`katex.min.css` lists `woff2` first and every supported browser prefers it; see
`src/llm_council/static/katex/FONTS.md`. The vendored copy is KaTeX 0.19.0 under the MIT
license (`LICENSE` in the same directory).

When `--file` is combined with `--question`, `--improve`, or `--prompt`, the
UTF-8 file is read
as context and the literal question is appended after it, separated by a blank line. The CLI
**renders the synthesis as Markdown on stdout by default**, regardless of whether saving
is enabled; headings, lists, emphasis, and code blocks are formatted for terminal reading.
Progress logs and errors/warnings go
to stderr. Errors exit with status 1; invalid CLI usage exits with status 2. `--no-save`
disables **all persistence**, even if other save flags or `.env` request it.
`SAVE_PROMPT=NONE` only disables prompt content, not response saving. Each normal CLI run
caches its latest synthesis in `.llm-council/latest-synthesis.md` (along with the alternating
user/assistant turn history in `.llm-council/latest-conversation.json`) for `--followup`; this is
separate from DuckDB/Markdown saving. `--no-save` suppresses updating that cache. A follow-up
requires an existing cached synthesis, replays the prior user/assistant conversation turns before
the new follow-up prompt so roles strictly alternate `user`/`assistant`, and then starts a normal
council run. `--copy` copies the exact synthesis
to the system clipboard (using `clip` on Windows, `pbcopy` on macOS, or `wl-copy`, `xclip`,
or `xsel` on Linux); it reports an error if no supported utility is available. `--export-markdown`
(or `llm_council.export_markdown()`) backfills stored DuckDB runs with non-empty syntheses into
`MARKDOWN_DIR` folders (or single synthesis files) without repeating API calls, skipping runs whose synthesis folder/file already
exists. See [troubleshooting](docs/operations/troubleshooting.md) for quota, exclusion, and missing-Markdown cases.

## Configuration

Copy [.env.example](.env.example) to `.env` in the **current working directory**.
Precedence: Python/CLI overrides > process environment > `.env` > defaults.
Loading `.env` does not mutate process environment. Relative output paths resolve
against the current working directory. `--env-file` / `env_file=` selects another file.

| Environment variable | Python setting / CLI flag | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | `api_key` (Python only) | required |
| `OPENROUTER_BASE_URL` | `base_url` (Python only) | `https://openrouter.ai/api/v1` |
| `LOG_LEVEL` | `log_level` / `--log-level` | `INFO` |
| `SAVE_PROMPT` | `save_prompt` / `--save-prompt` | `duckdb` |
| `SAVE_INITIAL_ANSWERS` | `save_initial_answers` / `--save-initial-answers` | `none` |
| `SAVE_SYNTHESIS` | `save_synthesis` / `--save-synthesis` | `none` |
| `DUCKDB_PATH` | `duckdb_path` / `--duckdb-path` | `council.duckdb` |
| `MARKDOWN_DIR` | `markdown_dir` / `--markdown-dir` | `council-output` |
| `JUDGE` | `judge` / `--judge` | `random` |
| `JUDGE_SCOPE` | `judge_scope` / `--judge-scope` | `free` |
| `JUDGE_PROMPT` | `judge_prompt` / `--judge-prompt` | unset (built-in backup) |
| `FREE_ROUTER_MODEL` | `free_router_model` (Python only) | empty; required for `free-router` |
| `AUTO_ROUTER_MODEL` | `auto_router_model` (Python only) | empty; required for `auto` |
| `ROUTER_MODELS` | `router_models` (Python only) | empty; comma-separated router IDs |
| `EXCLUDED_MODELS` | `excluded_models` / `--excluded-models` | empty (no built-in model exclusions) |
| `MAX_CONCURRENCY` | `max_concurrency` / `--max-concurrency` | `3` |
| `REQUEST_INTERVAL` | `request_interval` / `--request-interval` | `3.1` seconds |
| `REQUEST_TIMEOUT` | `timeout` / `--timeout` | `120` seconds/request |
| `MAX_RETRIES` | `max_retries` / `--max-retries` | `2` |
| `MAX_RETRY_WAIT` | `max_retry_wait` / `--max-retry-wait` | `60` seconds |
| `MAX_TOKENS` | `max_tokens` / `--max-tokens` | `2048` output tokens/call; a negative value omits the parameter and uses provider defaults |

Save modes are case-insensitive: `NONE`, `NO`, `NEITHER` (all disable), `DUCKDB`,
`MARKDOWN`, `BOTH`. Files and text prompts are saved to DuckDB by default; saving a
file copies its **contents**, not just its filename.

## Logging

Standard Python `logging` is rendered with Rich's `RichHandler` (colored levels and
readable console formatting) and reports discovery, model successes/failures, and retries
at `INFO` by default. At completion, the summary names the judge (without its vendor prefix
or `:free` suffix), counts successful member responses, and includes elapsed time, e.g.
`alpha synthesized 2 responses in 2 minutes 33 seconds:`. Set `LOG_LEVEL` in `.env` to
`DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` (case-insensitive). Environment variables override
`.env`; `--log-level` or Python `log_level=` overrides both:

```bash
uv run llm-council -q "Explain entropy" --log-level debug
uv run llm-council -q "Explain entropy" --log-level error > synthesis.md
```

```python
answer = query(prompt="Explain entropy", log_level="WARNING")
```

Failure and retry logs include the full provider error message and nested metadata, not just a status code. Transport failures, non-JSON error bodies, and generation errors retain their details too. The configured API key is redacted. Successful-run logs omit prompt/answer contents, but provider errors can echo submitted data; treat stderr/log files as sensitive. If all members fail, the raised error also lists each member's failure, even when warning logs are disabled. The package configures only the
`llm_council` logger, never the root logger or HTTP-library loggers. If the embedding
application already provides logging handlers, those are respected; otherwise one
Rich stderr handler is installed and reused across calls. Log level is package-wide, so
concurrent calls share the most recently configured level.

## Judge choices

| `judge` value | Selection and trade-offs |
|---|---|
| `random` | Random eligible catalog model; default scope is free. |
| `round-robin` | Rotate through sorted eligible IDs; thread-safe **within one Python process**. A new CLI process starts at the first ID. No hidden state file. |
| `free-router` | Use the ID configured in `FREE_ROUTER_MODEL`; actual routed model is recorded. Requires configuration. |
| `auto` | Use the ID configured in `AUTO_ROUTER_MODEL`; requires configuration and `judge_scope="all"` / `--judge-scope all` because it may cost money. |
| `most-intelligent` | First eligible model from OpenRouter's `intelligence-high-to-low` ranking with `min_intelligence_index=0`. Requires published benchmark coverage; raises an actionable error if none match. This is a benchmark heuristic, not a task-specific guarantee. |
| `largest-context` | Largest advertised context window; useful when combining many long answers. Not an intelligence ranking. |
| Exact model ID | Fixed, reproducible judge identity. Must be a text model in the current catalog; paid models require scope `all`. |

`JUDGE_SCOPE=all` broadens **judge selection only**. Initial council members are always
free. Judge selection is validated before sending prompts to members. Router selections
can vary, so inspect `result.judge`, not just `result.judge_requested`.

### External judge prompt
Both `query()` and `query_result()` expose the optional keyword-only parameter `judge_prompt: str | Path | None = None`. A supplied path overrides `JUDGE_PROMPT` from the environment or `.env` (and a supplied `Settings` object). Omit it or pass `None` to preserve configuration; pass `""` to select the built-in backup.

```python
from pathlib import Path
from llm_council import query

answer = query("Explain entropy", judge_prompt=Path("prompts/judge.md"))
```

Edit [prompts/judge.md](prompts/judge.md) and set `JUDGE_PROMPT=prompts/judge.md` in your local environment file. UTF-8 `.txt` and `.md` files (including BOM) work. Relative paths resolve from the current working directory, including when `--env-file` points elsewhere. Python `judge_prompt="path/to/prompt.md"` and CLI `--judge-prompt path/to/prompt.md` override the environment. The exact file contents are sent as the system message to every judge/fallback. Missing, unreadable, invalid UTF-8, or blank files fail before discovery, inference, or saving. Unset/blank `JUDGE_PROMPT` uses the retained `JUDGE_SYSTEM` backup.

No model IDs are built into application code. Ordinary strategies discover IDs from the live catalog. For routing, you can set `FREE_ROUTER_MODEL=openrouter/free` and/or `AUTO_ROUTER_MODEL=openrouter/auto` in your local environment file. These IDs are automatically excluded from individual council/ranked judge selection; `ROUTER_MODELS` lists additional router IDs. An exact configured router ID works as `JUDGE` too. Copy the recommended router/exclusion settings from [.env.example](.env.example) to preserve the previous exclusions; installing the package alone supplies no model-specific exclusions.

## Storage contract

### DuckDB: exactly two tables

One persistent `questions` table and one `responses` table serve every run:

- `questions`: `id VARCHAR PRIMARY KEY`, `created_at TIMESTAMPTZ`, `prompt TEXT`,
  `source_path VARCHAR`.
- `responses`: `id VARCHAR PRIMARY KEY`, `question_id VARCHAR REFERENCES questions(id)`,
  `created_at TIMESTAMPTZ`, `stage VARCHAR` (`initial` or `synthesis`),
  `requested_model VARCHAR`, `model VARCHAR`, `content TEXT`, `error TEXT`, `usage JSON`.

Each invocation has a new UUID (even for identical prompt text). Failed member/judge
calls are saved as error records when that stage's storage is enabled. Writes are
incremental: initial answers survive a failed synthesis. If responses are saved to
DuckDB but the prompt is not, a questions identity row is created with **NULL prompt
and source_path**. No prompt text is silently saved. No database is opened if every
DuckDB save mode is disabled. Do not run concurrent writer **processes** against the
same DuckDB file; use separate files or serialize runs.

Example analysis:

```sql
SELECT q.id, r.stage, r.model, r.content, r.error
FROM questions q JOIN responses r ON r.question_id = q.id
ORDER BY q.created_at, r.created_at;
```

### Markdown: one directory per prompt run (single file for synthesis-only)

Whenever synthesis succeeds, requested artifacts with actual non-empty response content
are created, all inside the same folder. Failed individual member or fallback-judge calls are
omitted from the folder (their error records can still be saved in DuckDB); if synthesis itself
fails (every member fails, or the judge and all fallbacks fail), **no Markdown folder or files are
created**:

```text
council-output/<first-six-prompt-words>-<YYYYMMDDThhmm>/
  prompt.md                          # if SAVE_PROMPT includes MARKDOWN
  initial-001-provider-model-free.md # one per non-empty member answer
  initial-002-other-model-free.md
  synthesis.md                       # if SAVE_SYNTHESIS includes MARKDOWN
```

When synthesis.md is the only Markdown file requested (no prompt.md or initial answers),
no subfolder is created. The synthesis is written directly under council-output using
the same slug/timestamp naming with an md extension:

```text
council-output/<first-six-prompt-words>-synthesis-<YYYYMMDDThhmm>.md
```

Examples:

```text
# Full run (prompt + initials + synthesis)
council-output/
  quick-brown-fox-jumps-over-lazy-20261014T1516/
    prompt.md
    initial-001-test-alpha-free.md
    initial-002-test-beta-free.md
    synthesis.md

# Synthesis-only run (SAVE_PROMPT=none, SAVE_INITIAL_ANSWERS=none, SAVE_SYNTHESIS=markdown)
council-output/
  some-good-question-here-is-synthesis-20261014T1451.md
```

Filenames are sanitized and prefixed with an index to avoid collisions. Folder names
use up to the first six meaningful prompt words (common articles/prepositions are
omitted) and a UTC minute timestamp; a numeric suffix disambiguates collisions. The
single-file synthesis name uses the same slug and timestamp with a -synthesis- marker
(e.g. council-output/some-good-question-here-is-synthesis-20261014T1451.md).
Original prompt text is preserved. Response files record requested and actual model
identities. Markdown artifacts are written only for non-empty response content. Unrequested
categories are omitted. No Markdown directory is made when saving is off or when synthesis fails.

If a model selected as judge already failed as a council member, synthesis tries a
model that succeeded during the same run instead. If the selected judge fails its own
synthesis request, the run likewise falls back to successful council models (excluding
any model that already errored). This avoids repeating a known-failing endpoint.

## Discovery, costs, privacy, and limits

- Every run fetches the live, paginated `/models` catalog with `input_modalities=text`
  and `output_modalities=text`. The free-model catalog request also sets `max_price=0`
  and `max_output_price=0`. Judge discovery with `judge_scope=all` repeats that catalog
  query without the price caps, and an explicit judge ID the free query hides is looked
  up unfiltered too, so paid judges stay selectable. Local validation requires zero prompt (`pricing.prompt`)
  and completion/output (`pricing.completion`) prices, plus zero optional
  request/reasoning/cache charges. Local validation also requires explicit text support
  in **both directions**, using modality arrays or the legacy `architecture.modality`
  summary when arrays are absent. Missing/null/malformed capabilities are skipped;
  multimodal models that also support text remain eligible. "Endpoints" here means
  **distinct catalog model IDs**, not every provider backing the same model. OpenRouter
  chooses the provider for each model request.
- Free detection requires explicitly zero prompt and completion/output prices, checks
  optional request, reasoning, and cache charges if present, and rejects conditional pricing
  overrides.
  Non-text models and configured routers are not council members. No `:free` suffix is fabricated.
- Restricted/harness-only endpoints must be configured in `EXCLUDED_MODELS`; there is no built-in ID blacklist. The example environment recommends `thinkingmachines/inkling-small:free` because it returned a harness-only 403 for this workflow. Tool support and agentic scores alone do not establish access restrictions.
- Configure unavailable/restricted endpoints via comma-separated `EXCLUDED_MODELS` in your local environment file, `--excluded-models provider/model:free,provider/other:free`, or Python `excluded_models=("provider/model:free",)`. Exclusions apply to members and judges, including configured routers and paid judges; explicit excluded judges are rejected before member calls. Overrides replace the entire list. DEBUG logs explain skipped models. OpenRouter's `auto`/`free-router` judges choose their underlying model
  server-side, so local exclusions cannot constrain those router decisions; use a
  catalog-selected or explicit judge if you need that guarantee.
- Free calls additionally constrain provider prompt/output pricing to zero to
  protect against catalog changes. A free catalog entry is not proof it will be
  available to your account; quotas, outages, and context limits can still fail calls.
- All member requests are attempted, with bounded concurrency and global request-start
  spacing. Retries honor `Retry-After` on 429/503 responses and on 402 responses whose
  `error.metadata.limit_source` is `openrouter_in_flight_budget`; other 402 errors are
  not retried. Waits longer than `MAX_RETRY_WAIT` fail rather than sleeping indefinitely
  or retrying too early. Other transient HTTP/API errors are retried, but ambiguous
  network timeouts are not, to avoid duplicate paid work. See [OpenRouter error handling
  and Retry-After guidance](https://openrouter.ai/docs/api_reference/errors-and-debugging)
  for retryable in-flight budget errors and non-retryable 402 insufficient-credit cases.
- `MAX_TOKENS` values below zero omit the `max_tokens` field entirely so OpenRouter can
apply the selected provider/model default; zero is invalid. The judge receives the
original question plus **all successful answers in full**.
  Nothing is silently truncated. Huge councils may exceed the judge's context; use
  `largest-context`, a larger explicit judge, or shorter member output. Token-limit
  truncation is treated as a failure; increase `MAX_TOKENS` when necessary.
- Default request spacing targets fewer than 20 starts/minute; actual per-minute/daily
  free quotas depend on your key/account and other clients. A council consumes one
  request per free model, plus one judge request and retries.
- `--no-save` prevents local storage, **not transmission**: the prompt goes to every
  selected model/provider and the judge. Candidate answers are treated as untrusted
  data by the judge instruction, but prompt-injection resistance is not guaranteed.
- Paid judges are opt-in; `auto` and scope `all` can incur charges. There is no total
  spending cap in this package. Secrets and default generated data are git-ignored.

Official references: [models and ranking](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties),
[chat completions](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion),
[free router](https://openrouter.ai/docs/guides/routing/routers/free-router),
[auto router](https://openrouter.ai/docs/guides/routing/routers/auto-router),
[rate limits](https://openrouter.ai/docs/api_reference/limits).

## Development

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests exercise real Python/CLI interfaces against a local mock OpenRouter HTTP server,
including DuckDB queries and Markdown artifacts; they do not require a key or spend
credits. Live provider quality, quota behavior, and routing need your configured key.
