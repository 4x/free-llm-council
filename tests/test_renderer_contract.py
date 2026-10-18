"""Contract tests for the shared Markdown + LaTeX renderer and its browser harness.

The renderer is JavaScript, and this project deliberately has no JavaScript test runner,
so the behaviour it must exhibit is exercised by
``tests/browser/markdown_renderer_harness.html`` (see ``tests/browser/README.md``). These
tests guard the parts Python can see: that the entry points and rules the UIs and reports
depend on are still present, that the risky rules stay scoped, and that neither the
renderer nor its harness ever reaches for something outside the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

from llm_council.report import MARKDOWN_RENDERER

HARNESS = Path(__file__).resolve().parent / "browser" / "markdown_renderer_harness.html"


def _renderer_source() -> str:
    return MARKDOWN_RENDERER.read_text(encoding="utf-8")


def test_renderer_exposes_the_document_entry_points() -> None:
    """Both consumers call these by name, so the names are part of the contract."""
    source = _renderer_source()

    for name in (
        "escapeHtml",
        "renderMathToHtml",
        "renderInlineMarkdown",
        "renderMarkdown",
        "sanitizeLatexDocument",
        "renderLatexDocument",
        "stripLatexComments",
        "shieldCode",
        "renderEnvironmentHtml",
    ):
        assert f"function {name}" in source, name


def test_renderer_typesets_bare_environments_kaTeX_supports() -> None:
    """Bare environment blocks are only attempted for names KaTeX can actually render."""
    source = _renderer_source()

    for name in ("equation", "align", "gather", "cases", "bmatrix", "array"):
        assert f'"{name}"' in source, name
    # eqnarray is the common real-world name for the aligned layout, and KaTeX has no
    # such environment, so it has to be mapped rather than attempted.
    assert '"eqnarray": "aligned"' in source
    assert '"displaymath": "aligned"' in source


def test_renderer_repairs_only_backslashes_that_cannot_be_row_breaks() -> None:
    r"""A real row break is followed by whitespace, "&", or an optional argument.

    Widening this rule would start eating the row breaks of a legitimately aligned
    environment, so the exclusion list is pinned here.
    """
    source = _renderer_source()

    assert "const DOUBLED_BACKSLASH_RE = /(\\\\{2,})" in source
    # The lookahead is the part that matters: whitespace, "&", and a "[" optional
    # argument all mean the run is a real row break and is left alone.
    assert "(?=[^\\s&\\[])" in source


def test_renderer_keeps_code_spans_out_of_the_rewriting_rules() -> None:
    """Literal code is the one part of stored text that must survive byte for byte."""
    source = _renderer_source()

    assert "CODE_SPAN_RE" in source
    assert "shieldCode(" in source
    assert "restoreCode(" in source
    # The shield has to be applied before the comment and backslash rules run.
    sanitizer = source.split("function sanitizeLatexDocument", 1)[1]
    assert sanitizer.index("shieldCode(") < sanitizer.index("stripLatexComments(")
    assert sanitizer.index("shieldCode(") < sanitizer.index("DOUBLED_BACKSLASH_RE")


def test_renderer_numbers_equations_and_resolves_references() -> None:
    """A reference prints the equation number, and stays visible when there is none.

    KaTeX consumes neither ``\\label`` nor ``\\ref``, so both are handled here: a label
    is recorded while its equation is numbered, and the reference is then rewritten to
    the number it points at. A reference that cannot be resolved keeps its target in
    brackets, because deleting it silently would be worse.
    """
    source = _renderer_source()

    # Only these commands point at a numbered equation. Citations have no numbering
    # source in a stored run, so they always keep their target.
    assert 'NUMBERED_REF_COMMANDS = ["ref", "eqref", "pageref"]' in source
    assert 'CITATION_COMMANDS = ["cite", "citep", "citet"]' in source
    for name in ("ref", "eqref", "pageref", "cite", "citep", "citet"):
        assert f'"{name}"' in source, name

    # An unresolvable reference still shows what it pointed at.
    assert re.search(r'return "\[" \+ label \+ "\]"', source)
    # \ref prints the bare number and \eqref the parenthesised one, as LaTeX does.
    assert re.search(r'command === "eqref" \? "\(" \+ number \+ "\)" : number', source)

    # Numbering has to run before the reference rule for a forward reference to resolve,
    # and the label map is created per rendered document: that is what makes the numbers
    # agree between the web UI and an exported report, which render a field separately.
    sanitizer = source.split("function sanitizeLatexDocument", 1)[1]
    labels = sanitizer.index("const labels = new Map()")
    numbered = sanitizer.index("numberEquations(out, labels)")
    resolved = sanitizer.index("resolveReferences(out, labels)")
    assert labels < numbered < resolved
    # The author's own \tag or \notag settles a number, so it must be checked here.
    assert "AUTHOR_TAG_RE.test(body)" in source
    assert "OWN_NUMBER_SUPPRESSED_RE.test(body)" in source

    # Labels and image files genuinely produce nothing, so those stay dropped.
    dropped = source.split("DROPPED_ARGUMENT_COMMANDS = [", 1)[1].split("]", 1)[0]
    assert '"label"' in dropped
    assert '"ref"' not in dropped


def test_renderer_and_report_never_reach_outside_the_repository() -> None:
    """The renderer ships inside standalone reports, so it may not reference a URL."""
    source = _renderer_source()
    markup = re.sub(r"//[^\n]*", "", source)

    assert re.findall(r"https?://", markup) == []
    assert "@import" not in markup


def test_browser_harness_is_self_contained_and_wired_to_the_vendored_engines() -> None:
    """The harness must run from a plain static file server, with no network access."""
    html = HARNESS.read_text(encoding="utf-8")

    assert re.findall(r"https?://", html) == []
    for asset in ("katex.min.css", "katex.min.js", "markdown_renderer.js"):
        assert asset in html, asset
    # It reports a machine-readable verdict rather than only printing a table.
    assert "window.harness = " in html
    assert "passed: results.length - failed.length" in html
