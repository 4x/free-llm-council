/* Shared Markdown + LaTeX renderer.
 *
 * Single source of truth for both consumers: the live web UI loads this file
 * from /static/markdown_renderer.js, and the HTML report export inlines the
 * same text. Keeping one implementation is deliberate - a second Markdown
 * renderer would drift from this one and reports would stop matching what the
 * UI shows.
 *
 * Requires window.katex to be present for LaTeX; degrades to literal source
 * (and to <code class="math-raw"> for inline math) when it is absent.
 *
 * Stored runs often hold a pasted LaTeX document rather than Markdown: bare
 * \begin{eqnarray}...\end{eqnarray} blocks with no $ delimiters, sectioning
 * commands, cross-reference commands, and (for text that came out of a PDF) command
 * names whose backslashes were doubled. renderMarkdown() typesets the bare math
 * environments wherever it meets them, and renderLatexDocument() (which runs
 * sanitizeLatexDocument() first) rewrites the rest of the document into Markdown for
 * every view of stored text. Neither guesses: whatever KaTeX cannot typeset is shown
 * as its literal source, never as quietly wrong math.
 */

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Private-use markers (NUL-delimited) that survive HTML escaping, so we can
  // pull code spans and LaTeX out of the text before Markdown transforms run
  // and splice the finished fragments back in afterwards.
  const INLINE_CODE_TOKEN = "\u0000CODEINLINE_";
  const INLINE_MATH_TOKEN = "\u0000MATHINLINE_";
  const DISPLAY_MATH_TOKEN = "\u0000MATHDISPLAY_";
  const ESCAPED_DOLLAR_TOKEN = "\u0000ESCDOLLAR\u0000";

  function katexReady() {
    return Boolean(window.katex) && typeof window.katex.renderToString === "function";
  }

  // Returns rendered HTML, or null when KaTeX is unavailable or the LaTeX is
  // unusable, so callers can fall back to showing the literal source. `strict`
  // turns parse errors into null instead of red error markup, which is how a
  // partly-broken document is detected before anything is shown.
  function renderMathToHtml(tex, displayMode, strict) {
    if (!katexReady()) return null;
    try {
      return window.katex.renderToString(tex, {
        displayMode: Boolean(displayMode),
        throwOnError: Boolean(strict),
        strict: false,
        trust: false,
      });
    } catch (_) {
      return null;
    }
  }

  // Replaces $...$ and \(...\) spans with tokens, pushing the raw LaTeX onto
  // `store`. Delimiters must hug their content ($x$, not $ x $) and a closing
  // $ followed by a digit is treated as currency, not math.
  function protectInlineMath(text, store) {
    let out = text.replace(/\\\(([\s\S]+?)\\\)/g, function(_, tex) {
      store.push(tex);
      return INLINE_MATH_TOKEN + (store.length - 1) + "\u0000";
    });
    out = out.replace(/\\\$/g, ESCAPED_DOLLAR_TOKEN);
    out = out.replace(/\$(?![\s$])([^$\n]*?[^\s$])\$(?!\d)/g, function(_, tex) {
      store.push(tex);
      return INLINE_MATH_TOKEN + (store.length - 1) + "\u0000";
    });
    return out.split(ESCAPED_DOLLAR_TOKEN).join("$");
  }

  function restoreInlineMath(html, store) {
    return html.replace(/\u0000MATHINLINE_(\d+)\u0000/g, function(_, idx) {
      const tex = store[Number(idx)];
      if (tex === undefined) return "";
      const rendered = renderMathToHtml(tex, false);
      if (rendered !== null) return rendered;
      return '<code class="math-raw">' + escapeHtml("$" + tex + "$") + "</code>";
    });
  }

  // Environments KaTeX typesets without $ delimiters, verified against the vendored
  // build. Names it has no equivalent for are mapped onto the closest one it has.
  const MATH_ENVIRONMENT_NAMES = [
    "equation", "equation*", "align", "align*", "alignat", "alignat*",
    "aligned", "alignedat", "gather", "gather*", "gathered", "split",
    "array", "cases", "matrix", "pmatrix", "bmatrix", "Bmatrix",
    "vmatrix", "Vmatrix", "smallmatrix",
  ];
  const MATH_ENVIRONMENT_ALIASES = {
    "eqnarray": "aligned",
    "eqnarray*": "aligned",
    "displaymath": "aligned",
    "math": "aligned",
    "subarray": "array",
  };

  // Non-math environments. Their wrappers become Markdown so the body reads as prose
  // instead of printing "\begin{thm}"; a label is the word the environment would
  // have typeset, and an empty label means the wrapper is simply dropped.
  const DOCUMENT_ENVIRONMENT_LABELS = {
    abstract: "Abstract", appendix: "Appendix", center: "",
    corollary: "Corollary", definition: "Definition", description: "",
    document: "", enumerate: "", example: "Example", figure: "Figure",
    flushleft: "", flushright: "", itemize: "", lemma: "Lemma",
    proof: "Proof", proposition: "Proposition", quotation: "", quote: "",
    remark: "Remark", table: "Table", thebibliography: "References",
    theorem: "Theorem", thm: "Theorem", titlepage: "",
  };

  // Layout commands KaTeX does not define. They contribute no content of their own, so
  // dropping them is what lets the surrounding math typeset at all.
  const DROPPED_LATEX_COMMANDS = [
    "centering", "bigskip", "medskip", "smallskip", "noindent", "newpage",
    "clearpage", "cleardoublepage", "pagebreak", "linebreak", "hfill", "par",
    "protect", "ignorespaces", "raggedright", "maketitle", "tableofcontents",
  ];

  // Commands whose arguments are labels or file names rather than content. A label
  // produces nothing in LaTeX either, and an image is represented by its caption.
  const DROPPED_ARGUMENT_COMMANDS = [
    "label", "index", "includegraphics", "bibliography", "bibliographystyle",
    "vspace", "hspace", "vskip", "hskip", "setcounter", "addtocounter",
    "footnotemark",
  ];

  // References are different: LaTeX replaces them with a number pointing at the target.
  // Where that number can be established (see numberEquations) the reference prints it;
  // where it cannot, the target is kept in brackets rather than deleted, so the reader
  // can still see that something was referenced.
  const NUMBERED_REF_COMMANDS = ["ref", "eqref", "pageref"];

  // Bibliography citations have no number to offer here: there is no bibliography to
  // number them from, so they always keep their target in brackets.
  const CITATION_COMMANDS = ["cite", "citep", "citet"];

  // LaTeX numbers every display equation in a document, in order, and \ref prints that
  // number. KaTeX does neither: it emits an empty tag placeholder and keeps no counter,
  // so an equation renders with no number at all. The number is therefore supplied here
  // as an explicit \tag, and the same number answers every reference to it - which is
  // what makes the reference and the equation it points at agree.
  //
  // Numbering is per rendered document, not per stored run: the web UI and an exported
  // report each render a field on its own, and a document that numbered itself from a
  // shared run-wide counter would show different numbers in the two views.
  const NUMBERED_ENVIRONMENT_NAMES = ["equation"];
  const LABEL_PATTERN = "\\\\label\\s*\\{([^{}]*)\\}";
  // An author who wrote \tag or \notag has already settled this equation's number (or
  // suppressed it), so the block is left exactly as written and does not consume a number.
  const AUTHOR_TAG_RE = /\\tag\s*\{/;
  const OWN_NUMBER_SUPPRESSED_RE = /\\(?:notag|nonumber)\b/;

  // Numbers every numbered environment in document order and records where each \label
  // points, so a reference can be answered whether it appears before or after its
  // target. Returns the rewritten text; `labels` receives label -> number.
  function numberEquations(source, labels) {
    const pattern = new RegExp(
      "\\\\begin\\{(" + NUMBERED_ENVIRONMENT_NAMES.join("|") + ")\\}" +
        "([\\s\\S]*?)\\\\end\\{\\1\\}",
      "g"
    );
    let number = 0;
    return source.replace(pattern, function(whole, name, body) {
      if (AUTHOR_TAG_RE.test(body) || OWN_NUMBER_SUPPRESSED_RE.test(body)) return whole;
      number += 1;
      const labelPattern = new RegExp(LABEL_PATTERN, "g");
      let found;
      while ((found = labelPattern.exec(body)) !== null) {
        const label = found[1].trim();
        // A duplicated label keeps its first number, as LaTeX's first definition wins.
        if (label && !labels.has(label)) labels.set(label, String(number));
      }
      return "\\begin{" + name + "}" + body + "\\tag{" + number + "}\\end{" + name + "}";
    });
  }

  // Replaces a reference with the number its label was given, in LaTeX's two forms:
  // \ref prints the bare number and \eqref the parenthesised one. A label defined in a
  // different stored field is not resolvable here, so it keeps the bracket marker.
  function resolveReferences(source, labels) {
    const pattern = new RegExp(
      "\\\\(" + NUMBERED_REF_COMMANDS.join("|") + ")\\s*\\{([^{}]*)\\}",
      "g"
    );
    return source.replace(pattern, function(whole, command, target) {
      const label = target.trim();
      const number = labels.get(label);
      if (number === undefined) return "[" + label + "]";
      return command === "eqref" ? "(" + number + ")" : number;
    });
  }

  // A run of backslashes that is not a row break is an escaping artefact of text that
  // passed through a JSON or PDF layer: "\\frac" is read by KaTeX as a row break
  // followed by the letters f-r-a-c, and a doubled "\\(" delimiter breaks the inline
  // math span it was meant to open. Both are wrong output that KaTeX does not report,
  // so they are repaired rather than displayed.
  //
  // A real row break is followed by whitespace, "&", or an optional argument ("\\[2pt]"),
  // so runs before anything else are the artefact. That keeps the row breaks in a
  // legitimately aligned environment intact.
  const DOUBLED_BACKSLASH_RE = /(\\{2,})(?=[^\s&\[])/g;

  const LATEX_SECTION_LEVELS = {
    part: 0, chapter: 1, section: 2, subsection: 3, subsubsection: 4,
  };

  // A literal percent ("~72% of samples") must survive, so a % only opens a comment at
  // the start of a line or directly after a token that ends LaTeX markup (} \ $ ]).
  // That keeps prose intact while still removing "\label{semi}%\begin{array}", which
  // would otherwise bring commented-out math to life.
  function stripLatexComments(source) {
    return String(source)
      .split(/\r?\n/)
      .map(function(line) {
        for (let i = 0; i < line.length; i++) {
          if (line.charAt(i) !== "%") continue;
          if (i > 0 && line.charAt(i - 1) === "\\") continue;
          const before = line.slice(0, i).replace(/\s+$/, "");
          if (before === "") return "";
          if ("}$\\]".indexOf(before.charAt(before.length - 1)) !== -1) return before;
        }
        return line;
      })
      .join("\n");
  }

  // Rewrite document-level LaTeX as the Markdown this renderer already understands.
  // Every rule targets a construct that would otherwise print literally; anything
  // unrecognised is left exactly as written.
  function sanitizeLatexDocument(source) {
    if (!source) return "";
    const codeStore = [];
    const labels = new Map();
    let out = shieldCode(String(source), codeStore);

    out = stripLatexComments(out);
    out = out.replace(DOUBLED_BACKSLASH_RE, "\\");

    // Numbers are assigned before the reference rule runs, so a forward reference
    // ("see \ref{defin}" above the equation it points at) resolves like any other.
    out = numberEquations(out, labels);

    out = out.replace(
      /\\(part|chapter|section|subsection|subsubsection)\*?\s*\{([^{}]*)\}/g,
      function(_, name, title) {
        const level = LATEX_SECTION_LEVELS[name] || 0;
        const text = title.trim();
        if (level === 0) return "\n\n" + text + "\n\n";
        return "\n\n" + "#".repeat(Math.min(level, 4)) + " " + text + "\n\n";
      }
    );
    out = out.replace(/\\(?:title|paragraph|subparagraph)\*?\s*\{([^{}]*)\}/g, function(_, title) {
      const text = title.trim();
      return text ? "\n\n**" + text + "**\n\n" : "\n\n";
    });
    out = out.replace(/\\caption\*?\s*\{([^{}]*)\}/g, function(_, caption) {
      return "\n\n*" + caption.trim() + "*\n\n";
    });

    out = out.replace(/\\(?:textbf|textsc|underline)\s*\{([^{}]*)\}/g, function(_, text) {
      return "**" + text.trim() + "**";
    });
    out = out.replace(/\\(?:emph|textit|textsl)\s*\{([^{}]*)\}/g, function(_, text) {
      return "*" + text.trim() + "*";
    });

    out = out.replace(
      new RegExp("\\\\(?:" + DROPPED_LATEX_COMMANDS.join("|") + ")\\b[ \\t]*", "g"),
      " "
    );
    out = out.replace(
      new RegExp(
        "\\\\(?:" + DROPPED_ARGUMENT_COMMANDS.join("|") + ")\\*?" +
          "(?:\\s*\\[[^\\]]*\\])*(?:\\s*\\{[^{}]*\\})*(?:\\s*\\[[^\\]]*\\])*",
        "g"
      ),
      " "
    );

    out = out.replace(
      new RegExp("\\\\(?:" + CITATION_COMMANDS.join("|") + ")\\*?" + "(?:\\s*\\[[^\\]]*\\])*" + "(?:\\s*\\{([^{}]*)\\})", "g"),
      function(_, target) {
        return "[" + target.trim() + "]";
      }
    );

    // Runs after the citation rule; the two command sets are disjoint.
    out = resolveReferences(out, labels);

    out = out.replace(/\\item\b[ \t]*(?:\[([^\]]*)\])?/g, function(_, label) {
      return "\n- " + (label ? "**" + label.trim() + "** " : "");
    });

    Object.keys(DOCUMENT_ENVIRONMENT_LABELS).forEach(function(name) {
      const label = DOCUMENT_ENVIRONMENT_LABELS[name];
      const start = new RegExp(
        "\\\\begin\\{" + name + "\\*?\\}(?:\\[[^\\]]*\\])?(?:\\{[^{}]*\\})?",
        "g"
      );
      const stop = new RegExp("\\\\end\\{" + name + "\\*?\\}", "g");
      out = out.replace(start, label ? "\n\n**" + label + ".** " : "\n\n");
      out = out.replace(stop, "\n\n");
    });

    return restoreCode(out.replace(/\n{3,}/g, "\n\n").trim(), codeStore);
  }

  // Fenced blocks and inline spans are literal code, so no rule above may reach inside
  // them: a Windows path or an escaped backslash in a code sample is not a LaTeX
  // artefact, and rewriting it would corrupt the one part of the text that is exact.
  const CODE_SPAN_RE = /(```[\s\S]*?```|`[^`\n]*`)/g;
  const CODE_SENTINEL = "\u0000LATEXCODE_";

  function shieldCode(source, store) {
    return source.replace(CODE_SPAN_RE, function(match) {
      store.push(match);
      return CODE_SENTINEL + (store.length - 1) + "\u0000";
    });
  }

  function restoreCode(source, store) {
    return source.replace(/\u0000LATEXCODE_(\d+)\u0000/g, function(_, index) {
      const original = store[Number(index)];
      return original === undefined ? "" : original;
    });
  }

  // Longest name first so "equation*" is preferred over "equation".
  function mathEnvironmentAlternation() {
    return MATH_ENVIRONMENT_NAMES.concat(Object.keys(MATH_ENVIRONMENT_ALIASES))
      .sort(function(a, b) { return b.length - a.length; })
      .map(function(name) { return name.replace(/\*/g, "\\*"); })
      .join("|");
  }

  // Typeset one bare environment block. A body written for equation or eqnarray often
  // carries row breaks or alignment tabs those environments reject, so the body is
  // retried as `aligned`, which accepts both. When nothing works the block keeps its
  // literal source: a reader must be able to tell the difference.
  function renderEnvironmentHtml(name, columnSpec, body, raw) {
    const mapped = MATH_ENVIRONMENT_ALIASES[name] || name;
    const attempts = [
      "\\begin{" + mapped + "}" + columnSpec + body + "\\end{" + mapped + "}",
      "\\begin{aligned}" + body + "\\end{aligned}",
    ];
    for (let i = 0; i < attempts.length; i++) {
      const rendered = renderMathToHtml(attempts[i], true, true);
      if (rendered !== null) return '<div class="math-display">' + rendered + "</div>";
    }
    // A rejected block is printed as its literal source, with the \tag this renderer
    // added for numbering removed first: the reader must see what was stored, not our
    // rewrite of it. Only a numeric tag is stripped, so an author's own is left alone.
    const literal = raw.replace(/\\tag\{\d+\}\s*(?=\\end\{)/, "");
    return (
      '<div class="math-display math-fallback">' +
      '<pre><code class="math-raw">' + escapeHtml(literal) + "</code></pre></div>"
    );
  }

  // Render stored text. A pasted LaTeX document is rewritten first, so comments,
  // sectioning, bare environments, and doubled backslashes all resolve; Markdown is
  // unaffected because none of those rules match it.
  function renderLatexDocument(text) {
    if (!text) return "<p><em>No content available.</em></p>";
    return renderMarkdown(sanitizeLatexDocument(text));
  }

  function renderInlineMarkdown(text) {
    const codeStore = [];
    const mathStore = [];
    let out = String(text ?? "").replace(/`([^`\n]+)`/g, function(_, code) {
      codeStore.push(code);
      return INLINE_CODE_TOKEN + (codeStore.length - 1) + "\u0000";
    });
    out = protectInlineMath(out, mathStore);
    out = escapeHtml(out);
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" style="color: var(--accent);">$1</a>');
    out = out.replace(/\u0000CODEINLINE_(\d+)\u0000/g, function(_, idx) {
      const code = codeStore[Number(idx)];
      return code === undefined ? "" : "<code>" + escapeHtml(code) + "</code>";
    });
    return restoreInlineMath(out, mathStore);
  }

  function renderMarkdown(md) {
    if (!md) return "<p><em>No content available.</em></p>";
    const codeBlocks = [];
    let fenced = String(md).replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, function(_, lang, code) {
      const idx = codeBlocks.length;
      const label = lang ? '<div style="font-family:var(--mono);font-size:10.5px;color:var(--text-dim);margin-bottom:4px;">' + escapeHtml(lang) + '</div>' : '';
      codeBlocks.push('<pre>' + label + '<code>' + escapeHtml(code.replace(/\n$/, "")) + '</code></pre>');
      return "\u0000CODEBLOCK_" + idx + "\u0000";
    });

    // Block-level math can span lines, so claim it before the text is split
    // into lines and isolate it on a line of its own. Each claim holds either TeX to
    // render later, or finished HTML for a bare environment, whose renderability is
    // settled here so an unusable block can fall back to its source.
    const displayMath = [];
    function claimDisplayMath(entry) {
      displayMath.push(entry);
      return "\n" + DISPLAY_MATH_TOKEN + (displayMath.length - 1) + "\u0000\n";
    }
    fenced = fenced.replace(/\$\$([\s\S]+?)\$\$/g, function(_, tex) {
      return claimDisplayMath({ tex: tex.trim() });
    });
    fenced = fenced.replace(/\\\[([\s\S]+?)\\\]/g, function(_, tex) {
      return claimDisplayMath({ tex: tex.trim() });
    });
    // A bare environment (no $ around it) gets its own display block too.
    fenced = fenced.replace(
      new RegExp(
        "\\\\begin\\{(" + mathEnvironmentAlternation() + ")\\}" +
          "(\\{[^{}]*\\})?([\\s\\S]*?)\\\\end\\{\\1\\}",
        "g"
      ),
      function(match, name, columnSpec, body) {
        return claimDisplayMath({
          html: renderEnvironmentHtml(name, columnSpec || "", body, match),
        });
      }
    );

    const lines = fenced.split(/\r?\n/);
    const html = [];
    let inUl = false;
    let inOl = false;
    let inQuote = false;
    let tableRows = [];

    function flushTable() {
      if (tableRows.length === 0) return;
      const parsedRows = tableRows.map(r =>
        r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim())
      );
      tableRows = [];
      if (parsedRows.length >= 2 && parsedRows[1].every(cell => /^:?-{2,}:?$/.test(cell))) {
        const headers = parsedRows[0];
        const bodyRows = parsedRows.slice(2);
        html.push(
          "<table><thead><tr>" +
          headers.map(h => "<th>" + renderInlineMarkdown(h) + "</th>").join("") +
          "</tr></thead><tbody>" +
          bodyRows.map(row =>
            "<tr>" + row.map(c => "<td>" + renderInlineMarkdown(c) + "</td>").join("") + "</tr>"
          ).join("") +
          "</tbody></table>"
        );
      } else {
        parsedRows.forEach(row => {
          html.push("<p>" + renderInlineMarkdown(row.join(" | ")) + "</p>");
        });
      }
    }

    function closeLists() {
      flushTable();
      if (inUl) { html.push("</ul>"); inUl = false; }
      if (inOl) { html.push("</ol>"); inOl = false; }
      if (inQuote) { html.push("</blockquote>"); inQuote = false; }
    }

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i];
      const trimmed = raw.trim();

      if (!trimmed) {
        closeLists();
        continue;
      }

      if (/^\|.*\|$/.test(trimmed)) {
        if (inUl) { html.push("</ul>"); inUl = false; }
        if (inOl) { html.push("</ol>"); inOl = false; }
        if (inQuote) { html.push("</blockquote>"); inQuote = false; }
        tableRows.push(trimmed);
        continue;
      } else {
        flushTable();
      }

      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        closeLists();
        html.push('<hr style="border:none;border-top:1px solid var(--border);margin:14px 0;" />');
        continue;
      }

      const codeMatch = trimmed.match(/^\u0000CODEBLOCK_(\d+)\u0000$/);
      if (codeMatch) {
        closeLists();
        html.push(codeBlocks[Number(codeMatch[1])]);
        continue;
      }        const displayMathMatch = trimmed.match(/^\u0000MATHDISPLAY_(\d+)\u0000$/);
        if (displayMathMatch) {
          closeLists();
          const entry = displayMath[Number(displayMathMatch[1])];
          if (entry && entry.html !== undefined) {
            html.push(entry.html);
          } else {
            const tex = (entry && entry.tex) || "";
            const rendered = renderMathToHtml(tex, true);
            html.push(
              rendered !== null
                ? '<div class="math-display">' + rendered + "</div>"
                : '<div class="math-display"><pre><code class="math-raw">' + escapeHtml(tex) + "</code></pre></div>"
            );
          }
          continue;
        }

      const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (headingMatch) {
        closeLists();
        const level = Math.min(headingMatch[1].length, 4);
        html.push("<h" + level + ">" + renderInlineMarkdown(headingMatch[2]) + "</h" + level + ">");
        continue;
      }

      const ulMatch = raw.match(/^\s*[-*+]\s+(.*)$/);
      if (ulMatch) {
        if (inOl || inQuote) closeLists();
        if (!inUl) { html.push("<ul>"); inUl = true; }
        html.push("<li>" + renderInlineMarkdown(ulMatch[1]) + "</li>");
        continue;
      }

      const olMatch = raw.match(/^\s*\d+[.)]\s+(.*)$/);
      if (olMatch) {
        if (inUl || inQuote) closeLists();
        if (!inOl) { html.push("<ol>"); inOl = true; }
        html.push("<li>" + renderInlineMarkdown(olMatch[1]) + "</li>");
        continue;
      }

      const quoteMatch = raw.match(/^\s*>\s?(.*)$/);
      if (quoteMatch) {
        if (inUl || inOl) closeLists();
        if (!inQuote) { html.push("<blockquote>"); inQuote = true; }
        html.push("<p>" + renderInlineMarkdown(quoteMatch[1]) + "</p>");
        continue;
      }

      closeLists();
      html.push("<p>" + renderInlineMarkdown(trimmed) + "</p>");
    }
    closeLists();

    let joined = html.join("\n");
    joined = joined.replace(/\u0000CODEBLOCK_(\d+)\u0000/g, function(_, idx) {
      return codeBlocks[Number(idx)] || "";
    });
    return joined;
  }

