#!/usr/bin/env python3
"""Create the next Bridge3R manuscript revision from sealed AIST MC artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manuscript", type=Path, required=True)
    parser.add_argument("--multicut-artifacts", type=Path, required=True)
    parser.add_argument("--destination-manuscript", type=Path, required=True)
    return parser.parse_args()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one replacement anchor in {path}: {old[:60]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source, artifacts, destination = (
        args.source_manuscript.resolve(),
        args.multicut_artifacts.resolve(),
        args.destination_manuscript.resolve(),
    )
    if not source.is_dir() or not artifacts.is_dir():
        raise FileNotFoundError("source manuscript or multi-cut artifacts are absent")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite manuscript revision: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial destination exists: {partial}")
    shutil.copytree(source, partial)
    try:
        artifact_destination = partial / "artifacts" / "aist_multicut_formal"
        shutil.copytree(artifacts, artifact_destination)
        supplement = partial / "sections" / "supp_d_results.tex"
        anchor = r"""\end{table*}

\subsection{EgoBody}"""
        addition = r"""\end{table*}

\subsection{AIST++ Repeated-Cut Scaling Test}
\label{app:aist-multicut}

We additionally test whether the fixed causal route composes across repeated
same-scene viewpoint switches. MC150-3 and MC150-4 are two independently
frozen 100-source official-test manifests, each retaining the 150-frame RGB
timeline while containing respectively two and three physical-camera cuts.
The runtime sees only the RGB stream. The detector processes every positive
from left to right; it neither receives the true cut count nor the camera
labels. Evaluation uses one first-shot Sim(3) anchor that is retained through
all later shots. Mean boundary quantities average the fixed evaluator-only
boundaries within each case, then use an unweighted case macro average.

Tables~\ref{tab:supp-aist-mc150-3} and
\ref{tab:supp-aist-mc150-4} retain all five causal internal routes and their
complete 100-case denominators. They are a repeated-event scaling and
component study, not a latency-equivalent comparison with the offline
PromptHMR CS150 pipeline. No result is pooled with CS150 because the number
and locations of camera transitions differ by construction.

\begin{table*}[t]
  \centering
  \caption{AIST++ MC150-3 repeated-cut test on a frozen 100-source official
  manifest. The first-shot Sim(3) is held fixed through two viewpoint cuts;
  all metrics are case-macro means. PA, Anchor, and mean seam-root are in mm;
  angular quantities are in degrees.}
  \label{tab:supp-aist-mc150-3}
  \scriptsize
  \resizebox{\textwidth}{!}{\input{artifacts/aist_multicut_formal/aist_mc150-3_formal_table}}
\end{table*}

\begin{table*}[t]
  \centering
  \caption{AIST++ MC150-4 repeated-cut test on a separate frozen 100-source
  official manifest. The first-shot Sim(3) is held fixed through three
  viewpoint cuts; all metrics are case-macro means.}
  \label{tab:supp-aist-mc150-4}
  \scriptsize
  \resizebox{\textwidth}{!}{\input{artifacts/aist_multicut_formal/aist_mc150-4_formal_table}}
\end{table*}

\subsection{EgoBody}"""
        replace_once(supplement, anchor, addition)
        experiments = partial / "sections" / "05_experiments.tex"
        old = """the multi-person table or presented as a latency-equivalent causal/offline
leaderboard.
"""
        new = """the multi-person table or presented as a latency-equivalent causal/offline
leaderboard. Two independent repeated-cut AIST++ manifests (MC150-3 and
MC150-4) are reported separately in Appendix~\\ref{app:aist-multicut}; they
test repeated causal-event composition without pooling a different transition
count into the CS150 comparison.
"""
        replace_once(experiments, old, new)
        protocol = partial / "tables" / "protocol_overview.tex"
        old = "AIST++ & 100 single-person official-test CS150 sources & supplementary causal/offline single-person transfer study \\\\"
        new = "AIST++ & 100 CS150 sources plus separate 100-source MC150-3/MC150-4 manifests & supplementary causal/offline transfer and repeated-cut scaling study \\\\"
        replace_once(protocol, old, new)
        readme = partial / "README.md"
        text = readme.read_text(encoding="utf-8")
        text = text.replace("# BRIDGE3R ICLR 2027 manuscript — v018", "# BRIDGE3R ICLR 2027 manuscript — v019", 1)
        text = text.replace(
            "This revision adds a sealed, 100-source AIST++ single-person CS150 result to\n",
            "This revision adds sealed, 100-source AIST++ repeated-cut MC150-3 and MC150-4 results alongside the existing CS150 study.\n\n"
            "The prior CS150 single-person result remains in the supplement. The new multi-cut tables retain complete frozen-test denominators and are framed as causal event-scaling/component evidence, rather than as a pooled or latency-equivalent offline comparison.\n\n"
            "The prior revision adds a sealed, 100-source AIST++ single-person CS150 result to\n",
            1,
        )
        readme.write_text(text, encoding="utf-8")
        changelog = partial / "CHANGELOG.md"
        changelog.write_text(
            """# Changelog

## v019_20260827_aist_multicut_formal

- Added sealed AIST++ MC150-3 and MC150-4 repeated-cut results. Each table is
  generated from a distinct frozen 100-source official-test manifest and is
  accompanied by a runtime/evaluator-separation campaign audit.
- Presented the new study as causal event-scaling and component evidence. It
  retains all causal internal rows and does not pool different cut counts or
  claim a latency-equivalent comparison with the offline PromptHMR CS150 row.

""" + changelog.read_text(encoding="utf-8").replace("# Changelog\n\n", "", 1),
            encoding="utf-8",
        )
        partial.rename(destination)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    print(destination)


if __name__ == "__main__":
    main()
