# Browser harnesses

The shared renderer (`src/llm_council/static/markdown_renderer.js`) is JavaScript, and this project deliberately has no JavaScript test runner: adding one would mean a Node toolchain for a single file. The behaviour that needs a real KaTeX build is therefore checked by opening a page.

## Running the renderer harness

From the repository root:

```powershell
uv run python -m http.server 8000
```

Then open <http://127.0.0.1:8000/tests/browser/markdown_renderer_harness.html>.

The page runs every assertion on load and prints `N passed, M failed` at the top, with a table of individual results and the rendered fixture document underneath. The same verdict is available to a script:

```powershell
# in the browser console of the harness page
window.harness
# { passed: 77, failed: [], errors: [] }
```

A non-empty `failed` list names the rule that regressed and shows the text it produced.

## What it covers

Every rule that exists because real stored runs needed it, so the harness doubles as the specification:

- literal percents survive while LaTeX comments are removed, including a comment hiding a whole environment
- bare `equation`, `eqnarray`, `bmatrix`, and `array` with a column spec all typeset
- a block KaTeX rejects falls back to its literal source exactly once, and renders no KaTeX
- doubled commands and doubled delimiters are repaired without touching real row breaks
- literal code spans keep their backslashes
- sectioning, theorem wrappers, itemize, labels, and references become Markdown
- every `equation` is numbered in order, `\ref` prints that number and `\eqref` prints it in parentheses, a forward reference resolves like any other, and the number beside the equation is the same one the reference shows
- an equation carrying the author's own `\tag`/`\notag`, or an `equation*`, is left alone and consumes no number
- numbering restarts per rendered document, so two fields never share a counter
- a reference to a label that does not exist in the document keeps its target in brackets, as does a citation
- `\text` inside math is left alone
- plain Markdown, `$$`, `$`, and escaped `\$` behave as before
- a whole fixture document renders with no fallback and no KaTeX error

Keep the fixtures inline and the harness free of network references: it must stay openable from a plain static file server, which `tests/test_renderer_contract.py` enforces.
