# Agent conduct

## Make life easy for human developer and user
**Save the humans time.**
Human monitoring, inspection, and intervention are important, but should be invoked only when necessary.

To the extent possible, the agent should do all activities needed to achieve the goal, including research, coding, scripting, documentation, and the work around what the human must do. For example, while the agent is not allowed to read or write `.env` files, it can prepare perhaps 80% in a file such as `draft.profile`, where the human only fills in actual secrets. This is especially useful when several associated `.env` files must be created within and across applications.

## Ask
The agent is encouraged to bring up questions for discussion **when direction is not clear or obvious**. Concisely present the options, pros, cons, expected consequences, implications, and recommendation as appropriate. If there is a strong recommendation, proceed with it rather than unnecessarily deferring the decision.

## Avoid disclaimers
I KNOW you're not a licensed/certified doctor/lawyer etc. I KNOW AI makes mistakes.

# Safety and permissions
- Access to `.env` files, account numbers, API tokens, and other secrets is normally off-limits unless the human explicitly grants permission.
- Do not touch anything outside this repository without explicit permission.
- Make sure you have permission before destructive actions.

# Project conventions

## Package and project management
This project is `uv`-managed. Use `uv` as the package and project manager:

1. Run Python commands via `uv run`.
2. Never set `PYTHONPATH` manually.
3. Never invoke `python` directly.

## Dates and timestamps
Always use Unix time internally.
For dashboards, logs, markdown, or anything a user is expected to normally interface, follow `TIME_ZONE` environment variable if available; otherwise system locale.

## Prefer enum
Over strings as appropriate.

# Code quality

## Coding style
- Follow relevant best practices; for example, all Python functions need docstrings and type hints.
- Avoid hard-coded or magic numbers; use constants and/or environment variables instead.
- Prefer f-strings to old-style (`printf`-style) string formatting.

## File organization
- Keep modules cohesive, reusable, and loosely coupled.

## Error handling

- Handle errors at every level.
- Provide user-friendly messages in UI code.
- Log detailed context server-side.
- Never silently swallow errors.

## Output and logging
Use `print` and `logging` appropriately:

- `print` for command-line output and final reports or summaries for users.
- `logging.info()` for confirming expected operation and other operational events.
- `logging.debug()` for detailed diagnostic information.
- `logging.warning()` for unexpected but handled situations.
- `logging.error()` for errors that need attention.

## Bug investigation
Identify the root cause rather than only treating symptoms:
1. What really caused this?
2. Why are the statements that caused the bug there, and what are they supposed to do?
3. What consequences would removing or altering those statements cause?

Keep fix small in scope, surgical in nature, touching only the code necessary to affect the change. Stress test for elegance and simplicity.

## Test modules
- Test modules should begin with `test_` and be saved in the `/tests` directory.

# Documentation and knowledge capture
Code should be well documented and easy for a newcomer to understand. Always update relevant `README.md` files and other pertinent documentation after adding a feature so the docs remain in sync with the code.

Capture durable project knowledge in `docs/`:

- Architecture decisions → `docs/architecture/`
- Operational failure modes → `docs/operations/troubleshooting.md`
- Roadmap and todos → `docs/planning/roadmap.md`
- Multi-step implementation plans → `docs/planning/plans/YYYY-MM-DD-slug.md`
- Raw data and schemas → `docs/reference/`
- Deprecated material → `docs/archive/`

## Versioning and Git workflow

- Work only on a dedicated branch, never directly on `main`.
- Commit work with a descriptive message.
- Dedicated feature or bugfix branches feed `main`; there are no release-candidate branches unless explicitly requested.
- Merge completed work into `main`.

## Runtime observability

- Put durable diagnostic detail in server-side logs and keep user-facing console messages concise and actionable.
- For changes that affect runtime behavior, document how to verify success on the runtime machine and which log events or health signals confirm it.

# Markdown

## Make use of Headers (#, ##, ###, ####, #####, ######) as appropriate

Subtitles should make use of headers rather than **bold**:

### use this style of subheaders

Which is preferred...
**Never use bold**: for what is actually subheaders

## Allow long lines

Do NOT insert artificial line breaks in the middle of paragraphs to enforce any character limit (reader uses word wrap).

## Prefer mermaid diagrams

Over ASCII diagrams or Unicode box-drawing art.

## No unnecessary whitespace

Such as blank lines after any # headers.

# Windows

- Both development and execution occur on Windows 10-11.
- Shell examples should prefer PowerShell.

# 
