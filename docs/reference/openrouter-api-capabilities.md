# OpenRouter API capabilities: coverage and gaps

Audit of what `llm-council` uses from the OpenRouter API versus what the platform actually offers. Written 2026-10-16 against the live OpenRouter API reference.

## What we use today

| Surface | Where | Notes |
| --- | --- | --- |
| `POST /chat/completions` | [client.py](../../src/llm_council/client.py) `complete()` | `model`, `messages`, `stream: false`, optional `max_tokens`, `provider.max_price` for free-only enforcement, and optional `tools: [{"type": "openrouter:web_search", "parameters": {"max_uses": 3}}]` for server-side web search |
| `GET /models` | [client.py](../../src/llm_council/client.py) `list_models()` | Filters: `input_modalities=text`, `output_modalities=text`, `max_price=0`, `max_output_price=0`, `limit`, `offset`; plus `sort=intelligence-high-to-low` and `min_intelligence_index=0` for the `most-intelligent` judge |
| Response `usage` | `complete()` → `ModelAnswer.usage` | Persisted to DuckDB; web UI shows `total_tokens` only |
| Error metadata | `_request()` | Special-cases 402 `openrouter_in_flight_budget` vs. other 402; honors `Retry-After` |

Nothing else. We never send attribution headers, never read per-generation metadata, never ask for structured output, never stream, and never inspect provider endpoints.

## Gaps worth closing

### 1. Per-generation metadata: `GET /generation?id=`

Every chat response includes an `id` (`gen-...`) which we currently discard. `GET /generation?id=<id>` returns cost, latency, generation time, finish reason, and the `provider_name` that actually served the request, plus prompt/completion/reasoning/cached token counts.

We store `model` but not `provider_name`, so the `Nvidia` 400 in [to-do.txt](../../to-do.txt) is only discoverable by reading the raw error. Fetching generation metadata would let us report per-member cost, latency, and provider, and would drive a "which free model is fastest / most reliable" leaderboard from our own runs.

Reference: [generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation)

### 2. Account and key state: `GET /key`

`GET /key` returns `limit`, `limit_remaining`, `limit_reset`, usage totals, and free-model daily request counts. This directly improves the diagnosis the README already documents for 402s: we could surface remaining free-tier quota before a run and after an in-flight-budget rejection, instead of telling the user to "top up credits".

`GET /credits` and `GET /activity` require a management key, so they are lower priority for a personal API key. The `rate_limit` field in `/key` is deprecated and should not be trusted.

Reference: [current API key](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key), [limits](https://openrouter.ai/docs/api_reference/limits)

### 3. Structured outputs for deterministic JSON

`_parse_improved_prompts()` in [council.py](../../src/llm_council/council.py) scrapes JSON out of free text: it strips Markdown fences, hunts for the outermost `[`/`]`, tries four wrapper keys, and falls back to splitting on numbered-list markers. That whole function exists because we ask for JSON as prose.

`response_format` with `type: json_schema` (`strict: true`) removes the guessing. Combine with `provider.require_parameters: true` so routing only picks endpoints that support schema-constrained output, and filter discovery with `supported_parameters=structured_outputs`.

Reference: [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)

### 4. Reasoning-model controls

Reasoning tokens are billed as output tokens and, on most providers, count against the request's `max_tokens`. That makes them a plausible contributor to our `exhausted MAX_TOKENS` path in `complete()`.

#### Suppressing returned reasoning is not the same as disabling reasoning

`reasoning.exclude: true` does **not** turn reasoning off. The model still reasons; the tokens are still generated, still billed, and still count against `max_tokens`. The parameter only omits the `reasoning` field from the response. It saves no tokens, no cost, and no latency.

For this codebase `exclude: true` is effectively a no-op: `complete()` reads only `choice["message"]["content"]` and never looks at `message.reasoning`, so returned reasoning already never reaches the judge. It cannot explain or fix anything we observe today.

`exclude: true` also makes the `exhausted MAX_TOKENS` failure harder to diagnose. When reasoning consumes the whole budget the response returns `finish_reason: "length"` with empty content; with reasoning excluded, the payload carries no text at all and the only signals left are `finish_reason` and `usage`.

#### What actually controls reasoning

- `reasoning.enabled: false` — disables reasoning (Anthropic `thinking: {type: "disabled"}` maps here).
- `reasoning.effort: "none"` — disables reasoning entirely. Accepted by OpenAI-style models (o-series, GPT-5) and Grok; roughly `minimal` ≈ 10% of `max_tokens`, `low` ≈ 20%, `medium` ≈ 50%, `high` ≈ 80%, `max` ≈ 95%.
- `reasoning.max_tokens: N` — caps the reasoning budget directly. Supported by Gemini thinking models, Anthropic reasoning models (Anthropic requires `max_tokens` to be strictly greater than the reasoning budget), and some Qwen thinking models.
- `reasoning.mode: "pro"` — routes to a deeper multi-pass reasoning variant at the same token rates but typically higher usage. GPT-5.6+ only.

`effort` and `max_tokens` are mutually exclusive. Prefer `reasoning` over the deprecated `include_reasoning`.

#### Discovering which models can be quieted

`GET /models` entries now carry a `reasoning` object for reasoning-capable models: `supported_efforts` (descending, or `null` when all gateway values are accepted), `default_effort`, `default_enabled`, `supports_max_tokens`, and `mandatory`. Non-reasoning models and routers (`openrouter/auto`, `openrouter/free`) omit it, and `default_effort: "none"` should be read as "off by default", not as a disable request.

`mandatory: true` means the model rejects `effort: "none"` — for those, capping with `max_tokens` or excluding the model is the only lever. This gives a concrete member-selection policy: filter out `mandatory` models when we want short deterministic answers, or send the lowest entry of `supported_efforts`.

#### Detecting reasoning that consumed the budget

`usage.completion_tokens_details.reasoning_tokens` is included in `usage.completion_tokens`, so `completion_tokens - reasoning_tokens` is the visible output. A near-zero difference means reasoning consumed the budget, not that the answer was long.

`complete()` now makes that distinction in the `exhausted MAX_TOKENS` message (see `_describe_length_stop()` in `client.py`), reporting token counts and the appropriate remedy:

- Reasoning used the whole budget → cap reasoning (`reasoning.max_tokens` / `reasoning.effort`) or drop the model; raising `MAX_TOKENS` cannot help.
- Reasoning used at least `REASONING_DOMINATED_SHARE` (50%) of the budget → reported as reasoning-dominated, because raising `MAX_TOKENS` mostly buys more reasoning.
- Visible output used the budget → increase `MAX_TOKENS`.
- No `completion_tokens_details` breakdown (or a malformed one) → reported as unknown rather than guessed.

When a partial answer accompanied the truncation, the message says it was discarded, since we still treat truncation as a failure.

Reference: [reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), [API parameters](https://openrouter.ai/docs/api_reference/parameters)

### 5. Server-side fallback routing via `models`

The judge fallback loop in `query_result()` re-issues a separate `POST` for each candidate and tracks `failed_ids` by hand to avoid spending on a model that already failed. Passing an ordered `models: [...]` array instead lets OpenRouter fail over inside a single request; the response `model` already reports which model answered, which `complete()` handles.

Reference: [model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks)

### 6. Richer provider preferences

We only send `provider.max_price`. Available and unused: `order`, `only`, `ignore`, `allow_fallbacks`, `require_parameters`, `data_collection`, `quantizations`, and `sort: {by: latency|throughput|price}`. Concretely: `data_collection: deny` for privacy, `sort` by latency for the judge to finish synthesis sooner, `ignore` providers with poor uptime. (`route` and the `:online` variant are deprecated — do not adopt them.)

Reference: [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection), [chat completions](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)

### 7. Provider endpoint data: `GET /models/{author}/{slug}/endpoints`

Returns each provider endpoint for a model with uptime, latency/throughput percentiles, context limits, quantization, and supported parameters. We select models but never providers, so a model with one flaky provider and one solid provider is all-or-nothing. This is the data needed to route around the kind of provider-specific 400 in [to-do.txt](../../to-do.txt).

Reference: [model endpoints](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model)

### 8. Streaming

`stream: true` (SSE) would give token-level progress in the web UI instead of only per-completed-model events, and avoids the `ReadTimeout` "already submitted" ambiguity in `_request()`. Caveat: with streaming, a failure can arrive as an SSE event after HTTP 200, so the client would need to inspect the stream body, not just the status.

### 9. Server tools (web search adopted; the rest unused)

`SEARCH_WEB` sends `tools: [{"type": "openrouter:web_search", "parameters": {"max_uses": 3}}]`, the current server-tool replacement for the deprecated `plugins: [{id: "web"}]` and `:online` surfaces. The model decides whether to search and may search up to `max_uses` times; `usage.server_tool_use.web_search_requests` reports how often it did. Discovery uses `supported_parameters` containing `tools`: the server tool needs tool calling, not provider-native search, because `engine: "auto"` falls back to Exa. Native search lives on the endpoint object as `native_tools["openrouter:web_search"]` and is only worth checking to predict which engine runs.

Search fees are billed in addition to token costs and are **not** capped by `provider.max_price`, so free models still incur search charges; `max_uses` bounds that exposure. Published rates per search request (up to 10 results included, then $0.001 per extra result; `max_results` defaults to 5, so our payload never pays the per-result surcharge):

| Engine | Per-request fee |
| --- | --- |
| Exa `instant` / `fast` / `auto` (the default fallback) | $0.007 |
| Exa `deep-lite` / `deep` | $0.012 |
| Exa `deep-reasoning` | $0.015 |
| Parallel `turbo` / `fast` | $0.001 |
| Parallel `basic` / `advanced` | $0.005 |
| Perplexity | $0.005 |
| Firecrawl | $0.005 per search + $0.0125 per returned result |
| Native (OpenAI, Anthropic, Google, Perplexity, SpaceXAI) | provider passthrough |

Free models therefore always run on Exa: none of the 16 models in the live `:free` catalog exposes `native_tools["openrouter:web_search"]`, and with `engine: "auto"` (our payload omits `engine`) a model without native search falls back to Exa `auto` at $0.007 per search. Our cap is 3 searches per request, so **$0.021 is the exact per-request worst case**; a full catalog-wide run (15 tool-capable members plus the judge) tops out near $0.34. The legacy plugin docs state the same conclusion directly: "Using web search will incur extra costs, even with free models".

The other server tools (`datetime`) and the remaining plugins (`file-parser` for PDF input, `response-healing`, an alternative to gap 3, and `context-compression`) are unused. Context compression is directly relevant to the judge failure that tells users to "try largest-context": it could let an ordinary judge digest a large council.

References: [server tools](https://openrouter.ai/docs/guides/features/server-tools), [web search tool](https://openrouter.ai/docs/guides/features/server-tools/web-search), [legacy plugins](https://openrouter.ai/docs/guides/features/plugins)

### 10. Attribution headers

`HTTP-Referer` plus `X-OpenRouter-Title` (legacy `X-Title`) register the app in OpenRouter rankings and analytics. A one-line change to the `httpx.Client` headers. The title alone does not create an app entry.

Reference: [app attribution](https://openrouter.ai/docs/app-attribution)

### 11. Discovery filters we do not use

`/models` also accepts `supported_parameters`, `context`, `category`, `q`, `arch`, `model_authors`, `providers`, `zdr`, `distillable`, `region`, and index ranges, with sorts including `throughput-high-to-low`, `latency-low-to-high`, `most-popular`, `top-weekly`, and `coding-high-to-low`. `GET /models/user` returns models already filtered by the account's provider preferences, privacy settings, and guardrails — a server-side version of our local `EXCLUDED_MODELS`. Examples: exclude models that do not actually support `max_tokens`, or bias member selection toward low-latency models.

### 12. Prompt caching and session affinity

`cache_control`, `prompt_cache_key`, and `prompt_cache_options` can cut latency when follow-up conversations resend identical history to every member and to the judge. Provider support varies and must be filtered via `supported_parameters`; free models may not support it, so this matters mainly for paid judges.

### 13. Error classification beyond 402

Our retry logic distinguishes only 402 in-flight budget from other 402s. OpenRouter also returns `limit_source: openrouter_key_limit` and `reason: weight_exceeds_budget` (retry is futile — reduce prompt or `max_tokens`), and 403 for guardrail/workspace budgets. Parsing these would directly satisfy the outstanding to-do "present full error message, not just status".

### 14. Off-mission but available

`POST /embeddings` with `GET /embeddings/models` could cluster or deduplicate member answers semantically instead of leaving it to the judge. `logprobs` / `top_logprobs` could quantify agreement between members. `POST /audio/transcriptions` and `/audio/speech` enable voice input/output. Presets (`GET /presets`, `@preset/slug`) could centralize judge configuration server-side, which cuts against the project's "configuration lives in `.env`" philosophy.

## Deprecated: do not adopt

`route`, the `:online` variant, `transforms`, `include_reasoning`, `usage.include`, and `stream_options.include_usage` are documented as deprecated or ineffective. Usage is now returned automatically on chat responses, including in the final streaming event.

## Suggested priority

1. Attribution headers — trivial, immediate.
2. `GET /generation` — unlocks cost/latency/provider reporting we already persist `usage` for.
3. `response_format` json_schema — deletes the fragile `_parse_improved_prompts` heuristics.
4. `reasoning` budget controls (`effort` / `max_tokens` / `enabled: false`) — the actual fix for reasoning consuming `MAX_TOKENS`; the truncation message now reports the reasoning/visible token split, so a reasoning-starved run is identified without extra instrumentation. Do **not** reach for `exclude`.
5. `GET /key` — concrete 402 diagnosis and free-tier visibility.
6. `models` fallback array + provider preferences — simplifies the judge loop.
