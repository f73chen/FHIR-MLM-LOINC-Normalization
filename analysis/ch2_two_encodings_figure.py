#!/usr/bin/env python
"""F-24: one lab Observation as the input of each model. The figure sets the FHIR-MLM tokens against
the text pieces that BioClinical ModernBERT reads, field by field (thesis Chapter 2).

The FHIR-native encoding is dense. It fits more lab observations into an 8,192-token window than the
text serialisation that BCM and Llama read. The numbers belong in the prose and caption. The figure
shows where the tokens go, which prose cannot show. Each row is one field of the Observation. FHIR-MLM
spends one token on it (two for a value and its unit). The text path writes the label of the field and
lets a subword tokenizer split the number or timestamp into pieces.

Reads results/analysis/ch2/two_encodings_tokens.json, written by analysis/ch2_two_encodings_tokens.py,
which runs the real dataset builders (pipeline/09, 10 and 11) on a synthetic patient holding the
running example (sodium, 139 mEq/L, range 133-145, the Observation that F-16 shows). Nothing is typed
in by hand. The tokens, the counts and which pieces carry a number are all read or derived from that
file. The FHIR-MLM lab-code token is drawn as [MASK], as the model receives it. The BCM leading-space
marker is not drawn, and its line-break pieces are drawn as ↵. The Llama piece count is printed for
the caption.

The paper style (FS.apply_paper) is a booktabs table at the thesis \\textwidth, with field names in
Computer Modern and tokens in CMU Typewriter. Each token is a thin-ruled box, and grey marks a token
that carries a number or a timestamp. There is no title and no colour. There is no bar chart either,
because the window counts read better in the caption.

Writes results/analysis/ch2/figures/ch2_two_encodings_v3.{png,pdf}.

  python analysis/ch2_two_encodings_figure.py
"""
from __future__ import annotations
import importlib.util
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath

HERE = Path(__file__).resolve().parent
_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)
FIG = HERE.parent / "results" / "analysis" / "ch2" / "figures"
TOKENS = HERE.parent / "results" / "analysis" / "ch2" / "two_encodings_tokens.json"

# Every length below is in points. The axes is drawn in points, so these are printed sizes.
SIZE = 9.0                  # field names, headers, counts
TOK = 7.5                   # token text
BOX_H = 1.55 * TOK          # token box height
BOX_PAD, BOX_GAP = 1.6, 1.8 # inside a box (each side), between boxes
LINE_GAP = 2.6              # between wrapped lines of boxes
ROW_PAD = 3.2               # above and below a row's content
PAD = 6.0                   # LaTeX's \tabcolsep
HEAVY, LIGHT = 0.8, 0.5     # booktabs rule widths
ABOVE, BELOW = 1.8, 2.9     # booktabs rule separations
SHADE = "#e0e0e0"
EDGE = "#5a5a5a"
FIELD_LW = 0.4

# The six text lines, by field (the formatter's order: Resource, Description, Code, time, value, range).
TEXT_FIELD = {"Resource": "Resource type", "Description": "Description", "Code": "Code",
              "EffectiveDateTime": "Time", "ValueQuantity": "Value and unit", "ReferenceRange": "Reference range"}
ROWS = ["Resource type", "Code", "Description", "Time", "Value and unit", "Reference range"]
NUMBER = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}|\d\.\d{3}e[+-]\d{2}")


def load():
    d = json.loads(TOKENS.read_text())
    f = d["fhir_mlm"]
    # FHIR-MLM emits: code, display, time, value bin + unit, low bin + unit, high bin + unit
    assert len(f) == 9 and f[0] == d["observation"]["code"]["coding"][0]["code"], f
    assert f[2].startswith("t_") and all(f[i].startswith("p_") for i in (3, 5, 7)), f
    fhir = {"Resource type": [], "Code": [("[MASK]", False)], "Description": [(f[1], False)],
            "Time": [(f[2], True)], "Value and unit": [(f[3], True), (f[4], False)],
            "Reference range": [(f[5], True), (f[6], False), (f[7], True), (f[8], False)]}
    text = {}
    for line, pieces in zip(d["bcm"]["lines"], d["bcm"]["pieces_by_line"]):
        field = TEXT_FIELD[line.split(":")[0]]
        # find each piece in the line, in order, to know which ones sit inside a number or the
        # timestamp. The tokenizer folds the space before [MASK] into the mask piece, so the pieces
        # cannot be concatenated back into the line.
        spans = [m.span() for m in NUMBER.finditer(line)]
        cells, pos = [], 0
        for p in pieces:
            if p == "Ċ":
                cells.append(("↵", False)); continue
            core = p.replace("Ġ", "")
            a = line.find(core, pos)
            assert a >= 0 and line[pos:a].strip() == "", (line, p, pos)
            pos = a + len(core)
            cells.append((core, any(a < hi and pos > lo for lo, hi in spans)))
        assert line[pos:] == "", (line, pos)
        text[field] = cells
    assert sum(len(v) for v in text.values()) == d["bcm"]["n"] == 77, d["bcm"]["n"]
    assert sum(len(v) for v in fhir.values()) == 9
    return fhir, text, d["llama"]["n"]


def build(write=True):
    fhir, text, n_llama = load()
    FS.apply_paper(SIZE)
    serif, mono = FontProperties(family=FS.SERIF, size=SIZE), FontProperties(family=FS.MONO, size=TOK)
    print("fonts:", Path(fm.findfont(serif)).name, Path(fm.findfont(mono)).name)
    tp = TextToPath()
    def tw(s, prop):
        return tp.get_text_width_height_descent(s, prop, ismath=False)[0]
    def boxw(s):
        return tw(s, mono) + 2 * BOX_PAD

    W = (FS.TEXTWIDTH - 2 * plt.rcParams["savefig.pad_inches"]) * 72
    head = ["Field", "FHIR-MLM", "n", "BioClinical ModernBERT (text)", "n"]

    # Rows that need two lines break at a set point inside the field. The reference range is low
    # then high in both columns, so the two columns mirror each other. FHIR-MLM breaks after its
    # first unit token, and the text breaks after the comma that ends "Low: ...". The text timestamp
    # puts its time-zone offset on the second line, which also shows what the offset alone costs.
    def segments(r, cells, side):
        texts = [t for t, _ in cells]
        if r == "Reference range":
            k = 2 if side == "fhir" else texts.index(",") + 1
        elif r == "Time" and side == "text":
            k = max(i for i, t in enumerate(texts) if t in ("-", "+"))   # the offset's sign
        else:
            return [cells]
        return [cells[:k], cells[k:]]
    def width_of(cells):
        return sum(boxw(t) + BOX_GAP for t, _ in cells) - BOX_GAP if cells else 0.0

    w_field = max(tw(s, serif) for s in ROWS + ["Total", head[0]])
    w_fhir = max(max(width_of(s) for r in ROWS for s in segments(r, fhir[r], "fhir")), tw(head[1], serif))
    w_n = max(tw("77", serif), tw("n", serif))
    x_field = PAD
    x_fhir = x_field + w_field + 2 * PAD
    x_n1 = x_fhir + w_fhir + 2 * PAD
    x_text = x_n1 + w_n + 2 * PAD
    x_n2 = W - PAD - w_n
    w_text = x_n2 - 2 * PAD - x_text

    def wrap(cells, width):
        lines, cur, used = [], [], 0.0
        for t, num in cells:
            w = boxw(t)
            if cur and used + BOX_GAP + w > width:
                lines.append(cur); cur, used = [], 0.0
            used += (BOX_GAP if cur else 0) + w
            cur.append((t, num, w))
        if cur: lines.append(cur)
        return lines

    wrapped = {r: [ln for s in segments(r, text[r], "text") for ln in wrap(s, w_text)] for r in ROWS}
    wrapped_fhir = {r: [ln for s in segments(r, fhir[r], "fhir") for ln in wrap(s, w_fhir + 1e-6)] for r in ROWS}
    def content_h(r):
        k = max(len(wrapped[r]), len(wrapped_fhir[r]), 1)
        return k * BOX_H + (k - 1) * LINE_GAP
    row_h = {r: content_h(r) + 2 * ROW_PAD for r in ROWS}
    head_h = SIZE * 1.45
    H = HEAVY + BELOW + head_h + ABOVE + LIGHT + BELOW + sum(row_h.values()) + ABOVE + LIGHT + BELOW + head_h + ABOVE + HEAVY

    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(0, H)

    def rule(y_top, lw):
        ax.plot([0, W], [y_top - lw / 2] * 2, color="black", lw=lw, solid_capstyle="butt")
    def label(x, y_mid, s, ha="left"):
        ax.text(x, y_mid - 0.33 * SIZE, s, ha=ha, va="baseline", fontsize=SIZE, family=FS.SERIF, color="black")
    def tokens(x0, y_top, lines):
        for i, line in enumerate(lines):
            y = y_top - i * (BOX_H + LINE_GAP) - BOX_H
            x = x0
            for t, num, w in line:
                ax.add_patch(Rectangle((x, y), w, BOX_H, fc=SHADE if num else "white", ec=EDGE, lw=FIELD_LW, joinstyle="miter"))
                ax.text(x + w / 2, y + BOX_H / 2 - 0.30 * TOK, t, ha="center", va="baseline", fontsize=TOK,
                        family=FS.MONO, color="black")
                x += w + BOX_GAP

    y = H
    rule(y, HEAVY); y -= HEAVY + BELOW
    mid = y - head_h / 2
    for s, x, ha in zip(head, (x_field, x_fhir, x_n1 + w_n, x_text, x_n2 + w_n), ("left", "left", "right", "left", "right")):
        label(x, mid, s, ha)
    y -= head_h + ABOVE; rule(y, LIGHT); y -= LIGHT + BELOW
    for r in ROWS:
        top = y - ROW_PAD
        first_mid = top - BOX_H / 2
        label(x_field, first_mid, r)
        if fhir[r]:
            tokens(x_fhir, top, wrapped_fhir[r])
        else:
            label(x_fhir, first_mid, "–")
        label(x_n1 + w_n, first_mid, str(len(fhir[r])), "right")
        tokens(x_text, top, wrapped[r])
        label(x_n2 + w_n, first_mid, str(len(text[r])), "right")
        y -= row_h[r]
    y -= ABOVE; rule(y, LIGHT); y -= LIGHT + BELOW
    mid = y - head_h / 2
    label(x_field, mid, "Total")
    label(x_n1 + w_n, mid, str(sum(len(v) for v in fhir.values())), "right")
    label(x_n2 + w_n, mid, str(sum(len(v) for v in text.values())), "right")
    y -= head_h + ABOVE; rule(y, HEAVY); y -= HEAVY
    assert abs(y) < 1e-6, y
    print(f"FHIR-MLM {sum(len(v) for v in fhir.values())} tokens | BCM {sum(len(v) for v in text.values())} | Llama {n_llama}")

    if write:
        FIG.mkdir(parents=True, exist_ok=True)
        print("wrote", FS.save(fig, str(FIG / "ch2_two_encodings_v3")))
    return fig


if __name__ == "__main__":
    build()
