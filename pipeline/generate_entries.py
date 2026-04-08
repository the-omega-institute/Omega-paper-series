#!/usr/bin/env python3
"""Generate .qmd entries for the Omega showcase site.

Reads paper metadata from automath's update_repo_readmes.py PAPERS list
and generates Quarto-compatible .qmd files for each paper.

Usage:
    python generate_entries.py                  # Generate all Gen 2 entries
    python generate_entries.py --gen1           # Generate Gen 1 entries from local dirs
    python generate_entries.py --dry-run        # Preview without writing
"""

import argparse
import textwrap
from pathlib import Path

ORG = "the-omega-institute"
SITE_ROOT = Path(__file__).parent.parent

# Gen 2 papers (from automath/tools/notebooklm-oracle/update_repo_readmes.py)
GEN2_PAPERS = [
    {
        "repo": "branch-cubic-regular-s4-closure-prym-ray-class",
        "title": "A Quartic Cover of 37a1 and Its Regular S4-Closure",
        "innovation": "For the conductor-37 elliptic curve, we carry out a complete explicit programme: constructing the regular S4-closure of a degree-4 map, decomposing the Jacobian via rational idempotents into a genus-2 Jacobian, a resolvent elliptic curve, and a Prym threefold.",
        "scope": "Arithmetic geometry",
        "keywords": ["elliptic curves", "Galois covers", "Jacobian decomposition", "Prym varieties", "ray class fields"],
        "target-journal": "Journal of Number Theory",
        "theorems": 39,
    },
    {
        "repo": "fibonacci-moduli-cross-resolution-arithmetic",
        "title": "Upper Fibers and Witness Covers for the Fibonacci Order-of-Apparition Map",
        "innovation": "We introduce witness covers for the fiber B_n, proving B_n is an upper set in the divisor lattice. Minimal generators are classified via a primitive-vs-ladder dichotomy for atomic prime-power birth moduli.",
        "scope": "Combinatorial number theory",
        "keywords": ["Fibonacci sequences", "rank of apparition", "divisor lattices", "multiplicative number theory"],
        "target-journal": "Research in Number Theory",
        "theorems": 29,
    },
    {
        "repo": "fibonacci-stabilization-sharp-threshold-conjugacy-nonlinearity",
        "title": "A Sharp Three-Window Threshold and Finite-Memory Conjugacy in Fibonacci Stabilization",
        "innovation": "A sharp threshold at window size m=3: for m>=3 the stabilized-window map is injective with image topologically conjugate to the full two-shift. The conjugacy transports the full thermodynamic formalism exactly.",
        "scope": "Symbolic dynamics and ergodic theory",
        "keywords": ["subshifts of finite type", "topological conjugacy", "thermodynamic formalism", "Fibonacci numeration"],
        "target-journal": "Journal of Physics A",
        "theorems": 70,
    },
    {
        "repo": "folded-rotation-histogram-certificates",
        "title": "Folded Histograms of Irrational Rotations: Sampling Certificates and Structural Mismatch to the Parry Measure",
        "innovation": "A two-stage audit framework for coarse-grained orbit data: certifying empirical folded histograms via star-discrepancy, TV, and relative-entropy bounds, then testing compatibility with the Parry equilibrium.",
        "scope": "Applied dynamical systems",
        "keywords": ["irrational rotations", "Sturmian sequences", "Parry measure", "Renyi divergence", "statistical auditing"],
        "target-journal": "SIAM Journal on Applied Dynamical Systems",
        "theorems": 46,
    },
    {
        "repo": "folded-rotation-histogram",
        "title": "Zeckendorf Folds, Sturmian Rigidity, and Parry Divergence on the Golden-Mean Shift",
        "innovation": "The canonical Zeckendorf fold on the Sturmian slice is bijective and realizes the higher-block presentation, giving an exact KL formula and sharp support-constrained Renyi minima in closed form.",
        "scope": "Ergodic theory and symbolic dynamics",
        "keywords": ["Sturmian subshifts", "Zeckendorf numeration", "golden-mean shift", "Parry measure", "Renyi divergence"],
        "target-journal": "Ergodic Theory and Dynamical Systems",
        "theorems": 46,
    },
    {
        "repo": "grg-shell-geometry-from-stationary-detector-thermality",
        "title": "Shell Geometry from Stationary Detector Thermality in Static KMS Spacetimes",
        "innovation": "Full click-record statistics of an Unruh-DeWitt detector define two codimension-one shells in static KMS spacetimes. In Schwarzschild the shell pair determines the mass parameter via a self-calibrating two-mode ratio law.",
        "scope": "Mathematical physics / QFT in curved spacetime",
        "keywords": ["Unruh-DeWitt detectors", "KMS states", "black hole thermodynamics", "symbolic dynamics", "Hawking radiation"],
        "target-journal": "General Relativity and Gravitation",
        "theorems": 122,
    },
    {
        "repo": "resolution-folding-core-symbolic-dynamics",
        "title": "Finite-Window Rigidity in Fibonacci Numeration",
        "innovation": "The Zeckendorf fold map is the normal-form map of a finite terminating confluent rewrite system, with a sharp block-bijection threshold at m=3. The right Fischer cover is identified as a suffix graph with 2^(m-1) states.",
        "scope": "Symbolic dynamics and formal language theory",
        "keywords": ["Zeckendorf expansion", "Fibonacci numeration", "shifts of finite type", "transducer theory", "Markov chains"],
        "target-journal": "Journal of Number Theory",
        "theorems": 68,
    },
    {
        "repo": "zeckendorf-streaming-normalization-automata-rairo",
        "title": "Canonical Zeckendorf Normalization and the Minimal Berstel Adder",
        "innovation": "The canonical low-to-high Zeckendorf normalization is non-subsequential with exact prefix-destruction index Delta(n)=n. The Berstel transducer is proved minimal with exact state complexity 10 via kernel separation.",
        "scope": "Formal language theory and automata",
        "keywords": ["Zeckendorf numeration", "transducer theory", "subsequential functions", "Fibonacci addition", "state complexity"],
        "target-journal": "RAIRO Theoretical Informatics and Applications",
        "theorems": 44,
    },
    {
        "repo": "zero-jitter-information-clocks-parry-gibbs-rigidity",
        "title": "Tilt Dynamics of Cylinder Information and the Parry Measure on the Golden-Mean Shift",
        "innovation": "Exponential tilts of cylinder information close within the one-step Markov family, making tilt dynamics globally linearizable. The Parry measure is characterized as the unique zero-jitter (constant Renyi spectrum) law.",
        "scope": "Ergodic theory and probability",
        "keywords": ["shifts of finite type", "Parry measure", "large deviations", "thermodynamic formalism", "Gibbs measures"],
        "target-journal": "Journal of Theoretical Probability",
        "theorems": 21,
    },
]

# Gen 1 papers (from existing repo directories)
GEN1_PAPERS = [
    {"dir": "NatComm-Static-Block-CA-Info-Laws-Observable-Languages", "title": "Static-Block Cellular Automata: Information Laws and Observable Languages", "target-journal": "Nature Communications", "scope": "Information theory and cellular automata"},
    {"dir": "JHEP-Einstein-Equations-Info-Geometric-Variational-Principle", "title": "Einstein Equations from Information-Geometric Variational Principle", "target-journal": "JHEP", "scope": "Information geometry and general relativity"},
    {"dir": "JMLR-Universal-Catastrophic-Safety-Undecidability", "title": "Universal Catastrophic Safety Undecidability and the Capability-Risk Frontier", "target-journal": "JMLR", "scope": "AI safety and computability theory"},
    {"dir": "PRA-Info-Rate-Circle-Dirac-Quantum-Walks", "title": "Information Rate Circle in Dirac Quantum Walks", "target-journal": "Physical Review A", "scope": "Quantum information and quantum walks"},
    {"dir": "PRD-Unified-Physical-Time-Scale", "title": "Unified Physical Time Scale: Scattering, Holography, Dirac-QCA", "target-journal": "Physical Review D", "scope": "Foundational physics"},
    {"dir": "PRL-Cosmological-Phase-Transitions-GW-Signatures", "title": "Cosmological Phase Transitions and Gravitational Wave Signatures from QCA", "target-journal": "Physical Review Letters", "scope": "Cosmology and gravitational waves"},
    {"dir": "PRL-Origin-Fermi-Dirac-Statistics-Topological-Impedance", "title": "Origin of Fermi-Dirac Statistics from Topological Impedance in QCA", "target-journal": "Physical Review Letters", "scope": "Quantum statistics"},
    {"dir": "SciPost-Self-Referential-Scattering-Birth-of-Fermions", "title": "Self-Referential Scattering and the Birth of Fermions", "target-journal": "SciPost Physics", "scope": "Particle physics from information theory"},
]


def slug(repo: str) -> str:
    return repo.replace("/", "-")


def gen2_qmd(paper: dict) -> str:
    keywords_str = ", ".join(paper["keywords"])
    return textwrap.dedent(f"""\
    ---
    title: "{paper['title']}"
    subtitle: "{paper['scope']}"
    date: 2026-04-07
    categories: [gen2, {paper['scope'].split()[0].lower()}, submitted]
    {target-journal}: "{paper['journal']}"
    status: "Submitted"
    repo: "{ORG}/{paper['repo']}"
    theorems: {paper['theorems']}
    featured: false
    description: "{paper['innovation'][:200]}..."
    ---

    ## Abstract

    {paper['innovation']}

    **Keywords:** {keywords_str}

    ## Key Metrics

    | Metric | Value |
    |--------|-------|
    | Machine-verified theorems | {paper['theorems']} |
    | Target journal | {paper['journal']} |
    | Status | Submitted |
    | Scope | {paper['scope']} |

    ## Video Explainer

    *Video coming soon --- NotebookLM pipeline ready, YouTube upload pending.*

    ## Links

    - [Full Paper (GitHub)](https://github.com/{ORG}/{paper['repo']})
    - [Lean 4 Proofs](https://github.com/{ORG}/automath/tree/dev-automation-integration/lean4)
    - [Automath Engine](https://github.com/{ORG}/automath)
    """)


def gen1_qmd(paper: dict) -> str:
    d = paper["dir"]
    return textwrap.dedent(f"""\
    ---
    title: "{paper['title']}"
    subtitle: "{paper['scope']}"
    date: 2025-12-01
    categories: [gen1, physics, {paper['journal'].split()[0].lower()}]
    {target-journal}: "{paper['journal']}"
    status: "In preparation"
    featured: false
    description: "Generation 1 paper targeting {paper['journal']}."
    ---

    ## About

    This is a Generation 1 paper from the Omega Project's quantum cellular automata era,
    exploring how foundational physics emerges from information-theoretic first principles.

    **Target journal:** {paper['journal']}

    **Scope:** {paper['scope']}

    ## Links

    - [Paper directory]({d}/)
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen1", action="store_true", help="Also generate Gen 1 entries")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    gen2_dir = SITE_ROOT / "science" / "gen2"
    gen1_dir = SITE_ROOT / "science" / "gen1"
    gen2_dir.mkdir(parents=True, exist_ok=True)
    gen1_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating Gen 2 entries ===")
    for p in GEN2_PAPERS:
        fname = slug(p["repo"]) + ".qmd"
        content = gen2_qmd(p)
        path = gen2_dir / fname
        if args.dry_run:
            print(f"  [DRY RUN] {path}")
        else:
            path.write_text(content)
            print(f"  [WROTE] {path}")

    if args.gen1:
        print("\n=== Generating Gen 1 entries ===")
        for p in GEN1_PAPERS:
            fname = slug(p["dir"]) + ".qmd"
            content = gen1_qmd(p)
            path = gen1_dir / fname
            if args.dry_run:
                print(f"  [DRY RUN] {path}")
            else:
                path.write_text(content)
                print(f"  [WROTE] {path}")

    total = len(GEN2_PAPERS) + (len(GEN1_PAPERS) if args.gen1 else 0)
    print(f"\nDone: {total} entries {'previewed' if args.dry_run else 'generated'}.")


if __name__ == "__main__":
    main()
