#!/usr/bin/env python3
"""Refactor Gen 2 paper .qmd pages with improved layout — badges, media section, infographic placement."""

import re
from pathlib import Path

SCIENCE_DIR = Path(__file__).parent.parent / "science" / "gen2"
ORG = "the-omega-institute"
REPO = "Omega-paper-series"
PAPERS_MEDIA_BASE = f"https://github.com/{ORG}/{REPO}/releases/download/papers-media-v1"

# Map repo slug → asset prefix (from update_media_links.py)
PREFIXES = {
    "branch-cubic-regular-s4-closure-prym-ray-class": "branch_cubic_regular_s4",
    "fibonacci-moduli-cross-resolution-arithmetic": "fibonacci_moduli_cross_resolution",
    "fibonacci-stabilization-sharp-threshold-conjugacy-nonlinearity": "fibonacci_stabilization_sharp_threshold",
    "folded-rotation-histogram-certificates": "folded_rotation_histogram_certificates",
    "folded-rotation-histogram": "folded_rotation_histogram",
    "grg-shell-geometry-from-stationary-detector-thermality": "grg_shell_geometry_from",
    "resolution-folding-core-symbolic-dynamics": "resolution_folding_core_symbolic",
    "zeckendorf-streaming-normalization-automata-rairo": "zeckendorf_streaming_normalization_automata",
    "zero-jitter-information-clocks-parry-gibbs-rigidity": "zero_jitter_information_clocks",
}


def extract_front_matter(content):
    """Return (front_matter_dict, body) from a .qmd file."""
    m = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not m:
        return {}, content
    fm_text, body = m.group(1), m.group(2)
    fm = {}
    for line in fm_text.split('\n'):
        if ':' in line and not line.startswith('-'):
            k, v = line.split(':', 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, body


def refactor(qmd_path: Path, repo_slug: str):
    prefix = PREFIXES[repo_slug]
    content = qmd_path.read_text()
    fm, _ = extract_front_matter(content)

    # Preserve original front matter exactly
    fm_match = re.match(r'^(---\n.*?\n---\n)', content, re.DOTALL)
    front_matter = fm_match.group(1) if fm_match else ""

    # Individual paper repo base URL for video/slides/podcast (stay on individual repos)
    paper_base = f"https://github.com/{ORG}/{repo_slug}/releases/download/media-v1"

    # Extract key data
    title = fm.get("title", "")
    subtitle = fm.get("subtitle", "")
    journal = fm.get("target-journal", "")
    status = fm.get("status", "Submitted")
    theorems = fm.get("theorems", "?")
    scope = fm.get("subtitle", "")  # subtitle is the scope

    # Extract abstract from body (first paragraph after ## Abstract)
    abstract_match = re.search(r'## Abstract\s*\n\n(.*?)\n\n', content, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    # Extract keywords
    keywords_match = re.search(r'\*\*Keywords:\*\*\s*(.+?)(?:\n|$)', content)
    keywords = keywords_match.group(1).strip() if keywords_match else ""

    # Build new body
    new_body = f'''
::: {{.paper-badges}}
[{journal}]{{.badge-journal}}
[{status}]{{.badge-status-submitted}}
[{theorems} Theorems]{{.badge-theorems}}
[{scope}]{{.badge-scope}}
:::

## Abstract

{abstract}

**Keywords:** {keywords}

## Infographic

![{prefix} Infographic]({PAPERS_MEDIA_BASE}/{prefix}_infographic.png)

## Video Explainer

{{{{< video {paper_base}/{prefix}_video.mp4 >}}}}

::: {{.media-downloads}}
[📊 Slides]({paper_base}/{prefix}_slides.pdf)
[🎧 Podcast]({paper_base}/{prefix}_podcast.wav)
[🎬 Video]({paper_base}/{prefix}_video.mp4)
[🖼️ Infographic]({PAPERS_MEDIA_BASE}/{prefix}_infographic.png)
:::

## Paper & Source

- [Full Paper (source in this repo)](../../papers/{repo_slug}/)
- [Machine-verified proofs (Lean 4)](https://github.com/{ORG}/automath/tree/dev-automation-integration/lean4)
- [Automath Engine](https://github.com/{ORG}/automath)
'''

    new_content = front_matter + new_body
    qmd_path.write_text(new_content)
    print(f"  ✓ {qmd_path.name}")


def main():
    for repo_slug in PREFIXES:
        qmd = SCIENCE_DIR / f"{repo_slug}.qmd"
        if qmd.exists():
            refactor(qmd, repo_slug)


if __name__ == "__main__":
    main()
