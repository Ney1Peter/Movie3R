# Shot3R ICLR 2027 manuscript — v072

This version contains the complete anonymous English manuscript. Set
`main.tex` as the root document. The source follows the official ICLR 2027
conference style and is configured for pdfLaTeX and BibTeX.

The scientific paper currently exceeds the final page target through the Conclusion.
This is an interim content-complete preview; page-limit compression has not yet
been applied. Required statements, references, and supplementary material
follow the main paper.

The v072 revision adds frozen traditional geometric-registration controls to
the main analysis and complete protocols, confidence intervals, viewpoint
strata, and failure diagnostics to the supplement. It also synchronizes the
supplement title with the final paper title. Existing checkpoints and prior
evaluation results are unchanged.

The 2026-09-18 typography pass makes method names inherit the official
Times-compatible Roman face, maps the local XeTeX bold face explicitly, and
sets every supplementary-table body to 9 pt. Wide tables are reflowed or
stacked instead of scaled down; table notes remain at 8 pt.

English passages are accompanied by Chinese `%` comments for editing. The
comments are invisible in the compiled PDF and require no CJK package.

See `BUILD.md` for compilation and validation details.
