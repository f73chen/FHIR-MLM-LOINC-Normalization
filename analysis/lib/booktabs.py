"""Booktabs tables, written as LaTeX and drawn in the paper style from the same rows. The drawing
uses serif text at 10 pt, rules only, booktabs rule widths and spacing, and right-aligned numbers.

A table has a `header` (one string per column), optional `groups` [(title, first_col, last_col)] that
span columns above the header, `body` rows of strings, and `right`, the set of right-aligned (numeric)
columns. With `total=True` the last body row is a total, set off by a midrule. `indent` is an optional
set of body-row indices whose first cell is indented by 1 em. These are sub-rows of the row above,
e.g. the splits that partition a pool. LaTeX uses `\\quad` and the drawing uses the same 1 em. `gaps`
is an optional set of body-row indices that start a new group of rows. LaTeX puts booktabs'
`\\addlinespace` (0.5 em) above each of them, and the drawing adds the same space. `widths` maps a column
to a fixed width in points. Its cells wrap at word boundaries, and a `$...$` span is never split.
LaTeX sets such a column as ragged-right `p{}`, which needs the array package. In a drawn fixed-width
column "\\n" forces a line break. For LaTeX the caller writes `\\newline` itself. A table too long
for one page is written as a `longtable` (`long=True`) and drawn in parts with `draw_parts()`, each
repeating the header. Strings may carry `<`, `≥`, `–` and mathtext (`$n$`, also inside text). `tex()`
translates them for LaTeX.

Used by the table scripts of thesis Chapters 2 and 4 to 6 (the ch2_, ch4_ and ch5_ table scripts and
v3_tables.py) and of the appendices (analysis/appendix/tables.py).
"""
from __future__ import annotations
import re

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

SIZE = 10.0                 # table text size, in points
ROW = 1.45 * SIZE           # row pitch (LaTeX \arraystretch about 1.2)
PAD = 6.0                   # LaTeX's default \tabcolsep, either side of every column
HEAVY, LIGHT = 0.8, 0.5     # booktabs \heavyrulewidth and \lightrulewidth (= \cmidrulewidth) at 10 pt
ABOVE, BELOW = 1.8, 2.9     # booktabs \aboverulesep (0.4ex) and \belowrulesep (0.65ex)
TRIM = 5.0                  # booktabs \cmidrulekern (0.5em): \cmidrule(lr) trims both ends by this
GAP = 0.5 * SIZE            # booktabs \defaultaddspace (0.5em): the space \addlinespace adds
LEAD = 1.2 * SIZE           # baseline to baseline inside a wrapped cell (LaTeX \baselineskip at 10 pt)
_TOKEN = re.compile(r"\$[^$]*\$\S*|\S+")    # a word, or a $...$ span with what is attached after it
_SUFFIX = re.compile(r"^(.*\S) ([(\[][^()\[\]]*[)\]])$")   # "0.0501 (0.0064)", "+2.13 [+2.03, +2.23]"


def _parts(s):
    """A cell's point estimate and its "(sd)" or "[CI]" suffix ("" if none)."""
    m = _SUFFIX.match(s)
    return (m.group(1), m.group(2)) if m else (s, "")


def split_cols(body, right):
    """The right-aligned columns whose point estimates would not line up if each cell were right-aligned whole.
    This happens when some cells carry a "(sd)" or "[CI]" suffix and the suffixes differ in width (a one-run model
    has none). Passed as `split`, such a column aligns the estimates and sets the suffixes after them."""
    width = _widthfn()[0]
    out = set()
    for k in right:
        sufs = [_parts(r[k])[1] for r in body]
        if any(sufs) and len({round(width(s), 3) if s else 0.0 for s in sufs}) > 1:
            out.add(k)
    return out


def _tex_cell(s):
    """Translate a plain-text cell (or caption) to LaTeX. The Unicode symbols the Chapter 6 tables use
    (minus sign, ±, ×, →, ≤, em dash) are translated too, so the .tex is plain ASCII and compiles under
    pdfLaTeX as is."""
    return (s.replace("%", r"\%").replace("<", "$<$").replace("≥", r"$\geq$").replace("≤", r"$\leq$")
            .replace("–", "--").replace("—", "---").replace("−", "$-$").replace("±", r"$\pm$")
            .replace("×", r"$\times$").replace("→", r"$\rightarrow$"))


def _head(header, groups, split=()):
    """The rows from the top rule to the header's midrule (repeated on each page of a longtable). A `split`
    column is two LaTeX columns (estimate, suffix), so spans and rule ends count physical columns."""
    n = lambda k: 2 if k in split else 1       # noqa: E731  (LaTeX columns per table column)
    p = lambda k: sum(n(j) for j in range(k))  # noqa: E731  (LaTeX columns before table column k)
    lines = [r"    \toprule"]
    if groups:
        cells, k = [], 0
        for title, a, b in groups:
            cells += [""] * (p(a) - p(k)) + [rf"\multicolumn{{{p(b) + n(b) - p(a)}}}{{c}}{{{_tex_cell(title)}}}"]
            k = b + 1
        cells += [""] * (p(len(header)) - p(k))
        lines.append("    " + " & ".join(cells) + r" \\")
        lines.append("    " + " ".join(rf"\cmidrule(lr){{{p(a) + 1}-{p(b) + n(b)}}}" for _, a, b in groups))
    heads = [rf"\multicolumn{{2}}{{r}}{{{_tex_cell(h)}}}" if k in split else _tex_cell(h) for k, h in enumerate(header)]
    return lines + ["    " + " & ".join(heads) + r" \\", r"    \midrule"]


def tex(header, body, groups=(), right=(), total=True, caption="", label="", comment="", indent=(), gaps=(),
        widths=None, long=False, split=()):
    """booktabs LaTeX. The table keeps its natural width, centred, like the drawn version. A `split` column
    (see split_cols) is `r@{\\ }l`: the estimates right-aligned, each suffix after one space."""
    widths = widths or {}
    spec = "".join(rf">{{\raggedright\arraybackslash}}p{{{widths[k] / 72:.2f}in}}" if k in widths
                   else r"r@{\ }l" if k in split else "r" if k in right else "l" for k in range(len(header)))
    gaps = {i for i in gaps if 0 < i < len(body) and not (total and i == len(body) - 1)}   # as draw()
    needs = r"% needs \usepackage{booktabs}" + (r", \usepackage{array}" if widths else "") \
        + (r", \usepackage{longtable}" if long else "")
    if long:
        lines = [f"% {comment}" if comment else "% generated", needs, rf"\begin{{longtable}}{{{spec}}}",
                 rf"    \caption{{{_tex_cell(caption)}}}\label{{{label}}} \\"]
        lines += _head(header, groups, split) + [r"    \endfirsthead"]
        lines += _head(header, groups, split) + [r"    \endhead", r"    \bottomrule", r"    \endfoot"]
    else:
        lines = [f"% {comment}" if comment else "% generated", needs,
                 r"\begin{table}[htbp]", r"  \centering", rf"  \caption{{{_tex_cell(caption)}}}", rf"  \label{{{label}}}",
                 rf"  \begin{{tabular}}{{{spec}}}"] + _head(header, groups, split)
    for i, r in enumerate(body):
        if total and i == len(body) - 1:
            lines.append(r"    \midrule")
        elif i in gaps:
            lines.append(r"    \addlinespace")
        cells = [" & ".join(_tex_cell(x) for x in _parts(c)) if k in split else _tex_cell(c) for k, c in enumerate(r)]
        if i in indent:
            cells[0] = r"\quad " + cells[0]
        lines.append("    " + " & ".join(cells) + r" \\")
    if long:
        return "\n".join(lines + [r"\end{longtable}", ""])
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _wrap(text, maxw, measure):
    """Greedy word wrap to `maxw` points. A word wider than the column breaks after its hyphens, as TeX
    breaks a compound word. A word with no hyphen stands on its own line."""
    units = []                                  # (piece, the glue before it)
    for w in _TOKEN.findall(text):
        if "-" in w.strip("-") and not w.startswith("$") and measure(w) > maxw:
            ps = re.findall(r"[^-]+-|[^-]+$", w)
            units += [(ps[0], " ")] + [(x, "") for x in ps[1:]]
        else:
            units.append((w, " "))
    lines, cur = [], ""
    for w, glue in units:
        cand = w if not cur else cur + glue + w
        if cur and measure(cand) > maxw:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    return lines + [cur] if cur or not lines else lines


def _space(width):
    return width("0 0") - width("00")


def _natural(header, body, groups, indent, widths, width, pad=PAD, split=()):
    """Column widths: the widest cell, a fixed width for a wrapped column, widened under a group title. A split
    column holds its widest estimate, a space and its widest suffix."""
    em = SIZE
    natural = [max(width(r[k]) for r in list(body) + [header]) if k not in widths else widths[k]
               for k in range(len(header))]
    for k in split:
        est, suf = zip(*(_parts(r[k]) for r in body))
        natural[k] = max(width(header[k]), max(map(width, est)) + _space(width) + max(width(s) for s in suf if s))
    if indent and 0 not in widths:
        natural[0] = max([width(header[0])] + [width(r[0]) + (em if i in indent else 0)
                                               for i, r in enumerate(body)])
    # a group header wider than its columns widens them equally
    for title, a, b in groups:
        span = sum(natural[a:b + 1]) + 2 * pad * (b - a)
        short = width(title) + 2 * TRIM - span
        if short > 0:
            for k in range(a, b + 1):
                natural[k] += short / (b - a + 1)
    return natural


def _widthfn():
    tp = TextToPath()
    prop = FontProperties(family="serif", size=SIZE)
    def width(s):                               # a whole cell: mathtext only when it starts with $
        return tp.get_text_width_height_descent(s, prop, ismath=s.startswith("$"))[0]
    def measure(s):                             # a wrapped line, which may mix text and $...$
        return tp.get_text_width_height_descent(s, prop, ismath="$" in s)[0]
    return width, measure


def draw(FS, header, body, path_stem, groups=(), right=(), total=True, indent=(), gaps=(), widths=None,
         natural=None, pad=PAD, split=()):
    """Draw the table at its natural width in the paper style and save PNG + PDF via FS.save.
    Returns (png path, width in inches). `natural` fixes the column widths (draw_parts uses it). `pad` is the
    column padding either side (LaTeX's \\tabcolsep), narrower for a table that would not fit otherwise. In a
    `split` column (see split_cols) the estimates are right-aligned and each suffix follows after a space."""
    FS.apply_paper(SIZE)
    widths = widths or {}
    width, measure = _widthfn()
    ncol = len(header)
    em = SIZE                                   # 1 em at the table's point size (LaTeX \quad)
    assert not set(split) & set(widths), "a split column cannot wrap"
    if natural is None:
        natural = _natural(header, body, groups, indent, widths, width, pad, split)
    sufw = {k: max(width(_parts(r[k])[1]) for r in body if _parts(r[k])[1]) for k in split}
    def cell(k, s, ind=False):                  # a cell's lines. In a wrapped column, "\n" starts a new line
        if k not in widths:
            return [s]
        return [ln for par in s.split("\n") for ln in _wrap(par, natural[k] - (em if ind else 0), measure)]
    head = [cell(k, h) for k, h in enumerate(header)]
    rows = [[cell(k, c, ind=(k == 0 and i in indent)) for k, c in enumerate(r)] for i, r in enumerate(body)]
    tall = lambda cells: ROW + (max(len(x) for x in cells) - 1) * LEAD   # noqa: E731
    left = [pad]
    for k in range(1, ncol):
        left.append(left[-1] + natural[k - 1] + 2 * pad)
    W = left[-1] + natural[-1] + pad
    n_light = (1 if groups else 0) + 1 + (1 if total else 0)   # cmidrule row, header midrule, total midrule
    gaps = {i for i in gaps if 0 < i < len(body) and not (total and i == len(body) - 1)}
    # without wrapped cells the height equals the formula below. This summation order keeps the drawings
    # of those tables byte-identical
    if not widths:
        H = (HEAVY + BELOW + (2 if groups else 1) * ROW + len(body) * ROW + n_light * (ABOVE + LIGHT + BELOW)
             + len(gaps) * GAP + ABOVE + HEAVY)
    else:
        H = (HEAVY + BELOW + (1 if groups else 0) * ROW + tall(head) + sum(tall(r) for r in rows)
             + n_light * (ABOVE + LIGHT + BELOW) + len(gaps) * GAP + ABOVE + HEAVY)
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(0, H)

    def rule(y_top, lw, x0=0.0, x1=W):        # a rule whose top edge is at y_top
        ax.plot([x0, x1], [y_top - lw / 2] * 2, color="black", lw=lw, solid_capstyle="butt")

    def put(k, y_bottom, lines, ind=False):    # y_bottom: the bottom of the row's first line
        x, ha = (left[k] + natural[k], "right") if k in right else (left[k] + (em if ind else 0), "left")
        for j, text in enumerate(lines):
            ax.text(x, y_bottom + 0.3 * ROW - j * LEAD, text, ha=ha, va="baseline", fontsize=SIZE,
                    color="black")

    y = H
    rule(y, HEAVY); y -= HEAVY + BELOW
    if groups:
        for title, a, b in groups:             # group headers, centred over their columns
            ax.text((left[a] + left[b] + natural[b]) / 2, y - ROW + 0.3 * ROW, title, ha="center",
                    va="baseline", fontsize=SIZE, color="black")
        y -= ROW + ABOVE
        for title, a, b in groups:             # \cmidrule(lr)
            rule(y, LIGHT, left[a] - pad + TRIM, left[b] + natural[b] + pad - TRIM)
        y -= LIGHT + BELOW
    for k, h in enumerate(head):
        put(k, y - ROW, h)
    y -= (ROW if not widths else tall(head)) + ABOVE; rule(y, LIGHT); y -= LIGHT + BELOW
    for i, r in enumerate(body):
        if total and i == len(body) - 1:       # midrule above the total
            y -= ABOVE; rule(y, LIGHT); y -= LIGHT + BELOW
        elif i in gaps:                         # \addlinespace
            y -= GAP
        for k, c in enumerate(rows[i]):
            if k in split:                      # the estimate ends where the column's widest suffix begins
                est, suf = _parts(body[i][k])
                x = left[k] + natural[k] - sufw[k]
                ax.text(x - _space(width), y - ROW + 0.3 * ROW, est, ha="right", va="baseline", fontsize=SIZE,
                        color="black")
                if suf:
                    ax.text(x, y - ROW + 0.3 * ROW, suf, ha="left", va="baseline", fontsize=SIZE, color="black")
                continue
            put(k, y - ROW, c, ind=(k == 0 and i in indent))
        y -= ROW if not widths else tall(rows[i])
    y -= ABOVE; rule(y, HEAVY); y -= HEAVY
    assert abs(y) < 1e-6, y
    return FS.save(fig, path_stem), W / 72


def draw_parts(FS, header, body, path_stem, parts, groups=(), right=(), total=True, indent=(), gaps=(),
               widths=None, split=()):
    """A long table drawn as several images, `parts` = [(first row, end row), ...], each with the header
    and the same column widths, saved as <stem>_part1, _part2, ... Returns (png paths, width in inches)."""
    FS.apply_paper(SIZE)
    widths = widths or {}
    assert parts[0][0] == 0 and parts[-1][1] == len(body) and all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
    assert not split, "split columns are not drawn in parts (each part would measure its own suffixes)"
    natural = _natural(header, body, groups, indent, widths, _widthfn()[0])
    pngs = []
    for n, (a, b) in enumerate(parts, 1):
        png, w = draw(FS, header, body[a:b], f"{path_stem}_part{n}", groups=groups, right=right,
                      total=total and b == len(body), indent={i - a for i in indent if a <= i < b},
                      gaps={i - a for i in gaps if a < i < b}, widths=widths, natural=natural)
        pngs.append(png)
    return pngs, w
