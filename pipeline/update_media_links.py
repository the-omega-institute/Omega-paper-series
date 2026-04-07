#!/usr/bin/env python3
"""Update Gen 2 .qmd files with media download links from GitHub releases."""

import re
from pathlib import Path

ORG = "the-omega-institute"
GEN2_DIR = Path(__file__).parent.parent / "science" / "gen2"

# Map repo slug to release asset prefix
REPOS = {
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


def update_qmd(qmd_path: Path, repo: str, prefix: str):
    base_url = f"https://github.com/{ORG}/{repo}/releases/download/media-v1"
    video_url = f"{base_url}/{prefix}_video.mp4"
    slides_url = f"{base_url}/{prefix}_slides.pdf"
    podcast_url = f"{base_url}/{prefix}_podcast.wav"

    content = qmd_path.read_text()

    # Update front matter - add media URLs
    if "video:" not in content:
        content = content.replace(
            'featured: false',
            f'featured: false\nvideo: "{video_url}"\nslides: "{slides_url}"\npodcast: "{podcast_url}"'
        )

    # Replace the "Video coming soon" section
    old_section = "*Video coming soon --- NotebookLM pipeline ready, YouTube upload pending.*"
    new_section = f"""{{{{< video {video_url} >}}}}

**Downloads:**

- [Slides (PDF)]({slides_url})
- [Podcast (WAV)]({podcast_url})
- [Video (MP4)]({video_url})"""

    content = content.replace(old_section, new_section)

    qmd_path.write_text(content)
    print(f"  [UPDATED] {qmd_path.name}")


def main():
    for repo, prefix in REPOS.items():
        qmd_path = GEN2_DIR / f"{repo}.qmd"
        if qmd_path.exists():
            update_qmd(qmd_path, repo, prefix)
        else:
            print(f"  [SKIP] {repo} - no .qmd file")

    print(f"\nDone: {len(REPOS)} entries updated with media links.")


if __name__ == "__main__":
    main()
