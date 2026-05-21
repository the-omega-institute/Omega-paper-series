# Distribution Boundary

This repository is an Omega academic content source. It is not the local publishing harness.

Use the local harness for publishing operations:

```text
../omega-broadcast-local
```

Use the generic toolkit for reusable platform automation:

```text
../broadcast-kit
```

## What Belongs Here

- academic source material
- paper-series generated media that should be archived with the content
- academic captions or thread drafts that are durable content assets
- distribution summaries and retrospectives for academic material
- release metadata for academic media assets

## What Does Not Belong Here

- Douyin/XHS/X auth state
- raw browser screenshots
- raw cookies or Playwright storage state
- global publish queues for multiple accounts
- unfiltered comments or private account feedback
- reusable publisher implementation code

## Agent Entry Point

When an agent starts in this repo and the task is about promotion, publishing, metrics, or feedback, read:

```text
.omega-source.yaml
../omega-broadcast-local/README.md
../omega-broadcast-local/config.yaml
```

Then decide whether the output should be written back here, to `../omega-ancient-texts-analysis`, or to `../broadcast-kit`.
