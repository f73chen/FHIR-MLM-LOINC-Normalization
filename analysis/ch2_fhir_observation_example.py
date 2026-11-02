#!/usr/bin/env python
"""F-16: one FHIR Observation resource, its fields colour-coded (thesis Chapter 2).

FHIR runs through the whole thesis, and this figure shows what a resource looks like. It renders one
lab Observation in the shape of the MIMIC-IV-on-FHIR `mimic-observation-labevents` profile. Each field
that the thesis discusses has its own colour.

The record is synthetic. Its shape (keys, systems, profile URL, extension URL, unit encoding) follows
the real export, but every value (ids, dates, numbers) is invented. Only one fact in it is real, and
it is public. MIMIC's concept map lists local item 50983 "Sodium" (fluid Blood, unit mEq/L) as
LOINC 2951-2 (Sodium SerPl-sCnc), which is also the worked example in Chapter 2. There is no patient
data. The caption must say "synthetic", because the image has no title that says so. The timestamp
carries a ±hh:mm offset because real ones do. It is the same running example that F-17 and F-24 use,
and F-24 tokenises this same timestamp.

The style is a listing as journals set one, with a typewriter face at its printed size inside a thin
square frame, like Walonoski 2018 (JAMIA, Listing 2). The paper style (FS.apply_paper) is CMU
Typewriter (Computer Modern, as in the LaTeX thesis) at 9 pt in black, drawn at the thesis \\textwidth
so it prints 1:1. There is no title and there are no callouts. The caption names each field by its
colour:

    local code       blue    "code": "50983"
    description      red     "display": "Sodium"
    value            green   "value": 139
    unit             pink    "unit": "mEq/L"
    reference range  brown   "referenceRange": [...]
    priority         purple  the mimic-lab-priority extension (url + valueString)

Every other field is black. The colours are used for text, so each one clears WCAG 4.5:1 on white.
They were also validated as a set over all pairs, for lightness band and chroma floor. The worst
normal-vision pair is red/brown at dE 17.6 (floor 15), and the worst colour-blind pair is blue/purple
at dE 9.4 deutan (target 8). They pass the colour-blind check without the JSON keys. The keys also
name each field, so identity does not rest on colour alone. The two weakest pairs are placed far apart
on the page. Coloured spans are bold, so they also stand out in greyscale.

Writes results/analysis/ch2/figures/ch2_fhir_observation_example.{png,pdf}.

  python analysis/ch2_fhir_observation_example.py
"""
from __future__ import annotations
import importlib.util
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

FIELD_COLOR = {
    "code": "#1f6fc5",      # blue
    "display": "#de2e0d",   # red
    "value": "#038342",     # green
    "unit": "#ad1c81",      # pink
    "range": "#87320f",     # brown
    "priority": "#5708cc",  # purple
}

# One list per printed line: (text, field). A field of None prints black. Monospace, so columns line up.
LINES = [
 [('{', None)],
 [('  "resourceType": "Observation",', None)],
 [('  "id": "a3f1c2e4-…",', None)],
 [('  "meta": { "profile": ["…/StructureDefinition/mimic-observation-labevents"] },', None)],
 [('  "status": "final",', None)],
 [('  "category": [{ "coding": [{ "code": "laboratory", … }] }],', None)],
 [('  "code": { "coding": [{', None)],
 [('      "system": "…/CodeSystem/mimic-d-labitems",', None)],
 [('      ', None), ('"code": "50983"', "code"), (',', None)],
 [('      ', None), ('"display": "Sodium"', "display"), (' }] },', None)],
 [('  "subject":   { "reference": "Patient/…" },', None)],
 [('  "encounter": { "reference": "Encounter/…" },', None)],
 [('  "effectiveDateTime": "2154-03-02T09:41:00-05:00",', None)],
 [('  "valueQuantity": { ', None), ('"value": 139', "value"), (', ', None), ('"unit": "mEq/L"', "unit"), (' },', None)],
 [('  ', None), ('"referenceRange": [{ "low": {"value": 133}, "high": {"value": 145} }]', "range"), (',', None)],
 [('  "extension": [{ ', None), ('"url": "…/StructureDefinition/mimic-lab-priority"', "priority"), (',', None)],
 [('                  ', None), ('"valueString": "ROUTINE"', "priority"), (' }]', None)],
 [('}', None)],
]

# Every length below is in points. The axes is drawn in points, so these are printed sizes.
SIZE = 9.0              # listing text
PITCH = 1.3 * SIZE      # line pitch
PADV = 5.0              # space between the frame and the first and last lines
FRAME = 0.5             # frame line width


def build(write=True):
    FS.apply_paper(SIZE)
    mono = FontProperties(family=FS.MONO, size=SIZE)
    print("font:", Path(fm.findfont(mono)).name)
    # Column width of the monospace font, from its unhinted glyph advances.
    cw = TextToPath().get_text_width_height_descent("M" * 100, mono, ismath=False)[0] / 100
    ncols = max(sum(len(t) for t, _ in segs) for segs in LINES)
    W = (FS.TEXTWIDTH - 2 * plt.rcParams["savefig.pad_inches"]) * 72
    inset = (W - ncols * cw) / 2          # the longest line sits centred in the frame
    assert inset >= 4, f"listing is too wide for the text width (inset {inset:.1f} pt)"
    first = PADV + 0.8 * SIZE              # first baseline, below the frame's top edge
    H = first + (len(LINES) - 1) * PITCH + 0.25 * SIZE + PADV

    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.add_patch(Rectangle((FRAME / 2, FRAME / 2), W - FRAME, H - FRAME, fill=False, ec="black",
                           lw=FRAME, joinstyle="miter"))
    # Every character is placed at its own column. With whole segments, the hinted glyph widths in the
    # PNG and the exact ones in the PDF drift apart over a long coloured span. The text after it (a
    # comma, a bracket) then lands with a gap in one format or overlaps in the other.
    for i, segs in enumerate(LINES):
        col = 0
        for txt, field in segs:
            for ch in txt:
                if ch != " ":   # va="baseline": glyphs of different heights still share one baseline
                    ax.text(inset + col * cw, H - first - i * PITCH, ch, family=FS.MONO, fontsize=SIZE,
                            va="baseline", ha="left", color=FIELD_COLOR.get(field, "black"),
                            weight="bold" if field else "normal")
                col += 1
    if write:
        FIG.mkdir(parents=True, exist_ok=True)
        print("wrote", FS.save(fig, str(FIG / "ch2_fhir_observation_example")))
    return fig


if __name__ == "__main__":
    build()
