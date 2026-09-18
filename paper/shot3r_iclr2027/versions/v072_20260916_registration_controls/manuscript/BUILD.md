# Build instructions

Set `main.tex` as the Overleaf root and select pdfLaTeX. The standard build is:

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

The archive includes the ICLR 2027 style, bibliography style,
`references.bib`, and a generated `main.bbl` fallback. It has no
machine-specific font or package dependency.

## v072 validation

- local validation engine: Tectonic 0.17 with BibTeX; the source retains the
  documented pdfLaTeX/BibTeX Overleaf route;
- document class: standard 10 pt `article` with `iclr2027_conference` and
  `times`, without custom margins, font-size overrides, or negative spacing;
- paper size: US Letter, 612 x 792 pt;
- main paper: pages 1--11, with the complete Conclusion on page 11;
- full English document: 41 pages; page-count compression is intentionally
  deferred;
- all fonts embedded; main prose is URW Nimbus Roman, the standard
  Times-compatible font selected by the LaTeX `times` package;
- local XeTeX validation explicitly maps Nimbus Roman regular, bold, italic,
  and bold italic; bold `Shot3R` entries therefore use Nimbus Roman Bold;
- all rendered supplementary-table bodies use one 9 pt size, with 8 pt
  reserved for table notes; no rendered supplementary table uses whole-table
  scaling;
- no fatal errors, undefined citations/references, or overfull boxes;
- only non-fatal underfull-box warnings remain;
- Figure 1 is below the anonymous author block and above the Abstract;
- all experiment floats are flushed before the Conclusion.

The Overleaf archive contains only files required by `main.tex`, plus this
README and build guide. It excludes Chinese-rendering sources, editable PPTX
files, logs, compiled PDFs, internal review documents, machine paths, and
machine-readable experiment audits that are not compilation dependencies.
