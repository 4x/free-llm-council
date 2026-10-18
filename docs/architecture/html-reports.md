# Self-contained HTML reports

`--export-html` turns a stored council run into one `.html` file that opens, archives, and prints with no network access and no llm-council install. This note records the decisions behind it, because two of them look redundant until you know why.

## One renderer, two consumers

Council answers are Markdown containing LaTeX. The web UI renders them in the browser; so do reports. Rather than write a second renderer, the rendering source lives in exactly one place:

```text
src/llm_council/static/markdown_renderer.js
```

- The web UI loads it as a blocking `<head>` script and calls `renderLatexDocument()` (or `renderMarkdown()`) directly.
- `report.py` inlines the same file's bytes into the exported document.

The single-source rule is load-bearing. A second Python Markdown renderer would be the obvious shortcut for a server-side export, and it would immediately diverge: syntax highlighting, table handling, the `$`-versus-currency heuristic, and code-fence rules would all drift, so reports would stop matching what the UI shows for the same run. `tests/test_report.py` asserts the inlined text is byte-identical to the asset file, so the two cannot silently separate.

The same logic applies to KaTeX: it is vendored under `static/katex/`, and the UI loads it by URL while reports inline it.

## Why reports need JavaScript

Pre-rendering to static HTML server-side would remove the JavaScript requirement, but only by duplicating the renderer — see above. So a report carries raw Markdown in a JSON payload plus the renderer, and typesets on open.

Two consequences are accepted deliberately:

- A JavaScript-disabled viewer sees the raw source. A `<noscript>` block carries the synthesis so the file is never blank.
- The payload is embedded in a `<script type="application/json">` element, so `<`, `>`, and `&` are written as `\u` escapes. Without that, a synthesis containing `</script>` could close the element and inject markup. Because the escapes are valid JSON, values round-trip unchanged.

## Why the fonts are inlined

`katex.min.css` declares each of its 20 font families with `woff2`, `woff`, and `ttf` sources. Only `woff2` is vendored (every supported browser prefers it — see `static/katex/FONTS.md`), and a report has no sibling `fonts/` directory, so `report.py` rewrites each `@font-face` `src` list down to a single inlined `data:` URI.

This is why a report is roughly 650 KB. That is the cost of being standalone, and it is preferred over a report that renders with fallback fonts.

Because the rewrite depends on the upstream stylesheet's exact shape, `inline_katex_stylesheet()` counts its substitutions and raises `StorageError` when the number is not 20. A re-vendored stylesheet that changes shape fails loudly at export time instead of producing a report that points at fonts it cannot embed.

## Idempotent, never destructive

`export_html()` skips a run when a file matching its name already holds byte-identical content, and any other name collision takes a `-N` suffix. It never overwrites an existing report. The practical consequence: changing report rendering produces a *new* file beside the old one rather than replacing it, so stale reports need deleting by hand.

## Related reading

- [LaTeX documents in stored runs](latex-documents.md) — what the shared renderer does with a pasted paper.
- [README](../../README.md) — user-facing behaviour and flags.
- [troubleshooting](../operations/troubleshooting.md) — report failure modes.
