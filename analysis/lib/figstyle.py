"""The figure style of the thesis: the model palette, the paper style (`apply_paper`) and `save`, which
writes each figure as PNG and PDF.

The four model colours pass the lightness band and the chroma floor. They stay distinguishable with
the common colour-vision deficiencies (worst pair dE 9.2, deutan) and with full colour vision (worst
pair dE 18.9). TF-IDF is magenta (#b8478f). Amber (#eda100) was too close to BCM's orange (dE 13.7,
below the floor of 15), and the two are harder still to tell apart in print or greyscale. Llama's
green (#1baf7a) has low contrast on white (2.74:1), so every figure that uses it also names the
model in a legend, a direct label or a table. A green dark enough to clear 3:1 would bring green and
orange to dE 5.0–5.3 for colour-blind readers.

Imported by most ch2_, ch4_ and ch5_ figure and table scripts, b3_figures.py, v3_figures.py,
v3_tables.py and analysis/appendix/tables.py.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- ink --------------------------------------------------------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# -- categorical: the four model families, in fixed order (never cycled) ----
COLOR = {
    "FHIR-MLM": "#2a78d6",
    "BCM": "#eb6834",
    "Llama-3.1-8B": "#1baf7a",
    "TF-IDF": "#b8478f",          # not amber: see the palette note above
}
ORDER = ["FHIR-MLM", "BCM", "Llama-3.1-8B", "TF-IDF"]

# The reference-range floor is a diagnostic, so it is drawn in a neutral grey that reads as a
# reference. That grey fails the categorical chroma floor, as intended.
FLOOR = "RefRange floor"
FLOOR_COLOR = "#52514e"

# -- sequential: one hue, light to dark (magnitude, never identity) --------
SEQ = ["#cfe0f6", "#9dc0ec", "#6b9fe1", "#2a78d6", "#1b5399"]

# Outcome shades for the four-way outcome of the off-the-shelf baselines (lib/oneshot.py), ordered from
# good to bad. They use one desaturated ramp instead of four hues, because the outcomes are ordered
# severity levels.
OUTCOME = {
    "correct": "#2a78d6",
    "valid_wrong": "#9aa7b4",
    "not_a_loinc": "#5c6672",
    "unparseable": "#2f353b",
}

MODEL_KEY = {"mlm": "FHIR-MLM", "bcm": "BCM", "llama": "Llama-3.1-8B", "tfidf": "TF-IDF"}


def apply():
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


# -- paper style -------------------------------------------------------------
# Thesis figures follow academic convention: no rounded "cards", pastel fills, grey secondary text,
# DejaVu Sans or titles inside the figure. Figures are set in the typeface of the thesis at their
# printed size:
#   - Computer Modern, because the LaTeX thesis is set in it. The faces are CMU (Computer Modern
#     Unicode, SIL OFL) from TeX Live: Serif for text, Bold Extended for bold (what LaTeX's \textbf
#     uses), Typewriter for code. TeX Live ships them as OpenType-CFF. matplotlib 3.10 embeds those in a
#     TrueType slot (/FontFile2), which gives an invalid PDF that strict readers reject. So they are
#     converted once to TrueType (fontTools' otf2ttf recipe) into a gitignored cache and embedded as
#     TrueType. Maths uses the Computer Modern that ships with matplotlib ("cm").
#   - Fallbacks per glyph, for the few characters CMU lacks: STIX (≥, ✓) for serif, FreeMono (↵) for
#     code. matplotlib only falls back across the families in a list, so figure code passes the lists
#     SERIF or MONO in place of the bare "serif" / "monospace" aliases.
#   - TEXTWIDTH is \textwidth in the University of Waterloo thesis template, so a full-width figure is
#     drawn 1:1 and its text prints at the size it was set in.
#   - Black text and thin black rules. The caption carries the title and every explanation.
#   - Fonts embedded as TrueType (pdf.fonttype 42). Publishers reject Type 3 fonts.
TEXTWIDTH = 6.375   # inches
SERIF = ["CMU Serif", "STIXGeneral"]
MONO = ["CMU Typewriter Text", "FreeMono"]
# STIX (a Times design) replaces Computer Modern for generators whose output is measured in it. The
# appendix tables (analysis/appendix/tables.py) size their LaTeX columns from STIX metrics, and the
# copies of those tables in the thesis are fitted to them. They set FONT = "stix" before drawing.
FONT = "cm"
STIX_SERIF = ["STIXGeneral", "Liberation Serif", "Times New Roman"]
STIX_MONO = ["FreeMono", "Courier New", "Liberation Mono"]
CM_OTF = locations.CMU_FONTS
CM_TTF = Path(__file__).resolve().parent / "fonts_cm"          # gitignored; rebuilt from CM_OTF
CM_FACES = {"cmunrm": None, "cmunbx": "CMU Serif Bold", "cmunti": None, "cmunbi": "CMU Serif Bold Italic",
            "cmuntt": None, "cmuntb": None, "cmunit": None}   # renamed: matplotlib reads "Extended" as a wider face


def _cff_to_ttf(src, dst, full_name=None):
    """fontTools' otf2ttf snippet, which converts cubic CFF outlines to quadratic glyf (error <= 1 unit of 1000)."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    font = TTFont(src, recalcTimestamp=False)      # keep the source's timestamp: a rebuild is byte-identical
    order, glyphs = font.getGlyphOrder(), font.getGlyphSet()
    font["loca"], font["glyf"] = newTable("loca"), newTable("glyf")
    glyf = font["glyf"]
    glyf.glyphOrder, glyf.glyphs = order, {}
    for name in order:
        pen = TTGlyphPen(glyphs)
        glyphs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    for name, g in glyf.glyphs.items():
        if hasattr(g, "xMin"):
            font["hmtx"][name] = (font["hmtx"][name][0], g.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion, maxp.maxZones = 0x00010000, 1
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions"):
        setattr(maxp, k, 0)
    maxp.maxComponentElements = max((len(getattr(g, "components", [])) for g in glyf.glyphs.values()), default=0)
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    if full_name:
        for rec in font["name"].names:
            if rec.nameID == 4:
                rec.string = full_name
    font.sfntVersion = "\000\001\000\000"
    font.save(dst)


def _register_cm():
    from matplotlib import font_manager
    CM_TTF.mkdir(exist_ok=True)
    for stem, full_name in CM_FACES.items():
        ttf = CM_TTF / f"{stem}.ttf"
        if not ttf.exists():
            if not (CM_OTF / f"{stem}.otf").exists():
                raise FileNotFoundError(f"{CM_OTF / stem}.otf: Computer Modern needs TeX Live's cm-unicode fonts")
            _cff_to_ttf(CM_OTF / f"{stem}.otf", ttf, full_name)
        font_manager.fontManager.addfont(str(ttf))


def apply_paper(size=10):
    """Academic style for thesis figures. `size` is the printed point size."""
    if FONT == "cm":
        _register_cm()
    serif, mono = (SERIF, MONO) if FONT == "cm" else (STIX_SERIF, STIX_MONO)
    plt.rcParams.update({
        "font.family": serif, "font.serif": serif, "font.monospace": mono, "font.size": size,
        "mathtext.fontset": "cm" if FONT == "cm" else "stix",
        "axes.titlesize": size, "axes.labelsize": size, "legend.fontsize": size - 1,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.0,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finish(ax):
    """Drop the top/right spines so the axes stay in the background."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, path_stem):
    """Save the figure as PNG (to look at) and PDF (for the thesis)."""
    for ext in ("png", "pdf"):
        # no creation date in the PDF, so re-rendering unchanged content leaves the file byte-identical
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(f"{path_stem}.{ext}", bbox_inches="tight", metadata=meta)
    return f"{path_stem}.png"
