# Build instructions

Set `main.tex` as the Overleaf root and use pdfLaTeX with BibTeX:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Equivalently:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The archive has no machine-specific TeX dependency. It includes `references.bib`, the ICLR bibliography style, and `main.bbl` as a fallback. Chinese translations are ordinary LaTeX comments and require no CJK package.

## v066 validation

- anonymous ICLR 2027 style and US-Letter output;
- one compilation root for the paper, statements, references, and supplement;
- 34 pages in the complete local Tectonic build;
- scientific paper occupies pages 1--9 and the complete Conclusion remains on page 9;
- statements begin on page 10, references on page 11, and appendix contents on page 16;
- Figure 1 is fixed below the anonymous author block and above the Abstract, which remains complete on page 1;
- no fatal errors, undefined references/citations, or overfull boxes;
- only non-fatal underfull-box warnings in the local TeX environment;
- all twelve confirmed v065 result-audit decisions are integrated without changing retained measurements;
- the main alignment--future-state table reports all 129 EgoBody cases;
- viewpoint statistics use 90 EgoHumans cases, 27 capture clusters, and 50,000 bootstrap draws;
- association-cue results use all 88 Harmony4D cases;
- the transition-count control uses a fixed 300-frame timeline and 0/1/3/5 transitions;
- the existing seven-method full-system runtime protocol remains distinct from the controlled reconstruction-path timing;
- no rendered drafting placeholders, machine paths, usernames, or old reader-facing Bridge3R/Strict Human3R terms;
- the bibliography contains 65 cited entries spanning the six relevant reviewer communities;
- the final anonymous archive is recompiled after clean extraction before delivery.

The v066 paper retains the supplied v9 teaser, minimally corrected pipeline, and compact shot-alignment module. The teaser uses a non-floating opening-page layout. Metadata-cleared editable decks remain in the working version and are excluded from the anonymous archive.

The editable PPTX/SVG files and private machine-readable audits remain in the working version, not in the anonymous Overleaf archive.
