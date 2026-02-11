---
name: diagram-recreate
description: Recreate diagrams from images in a consistent article style
---

## What I do

- Extract structure from a diagram image (nodes, edges, groups, annotations)
- Normalize labels and simplify layout
- Produce a clean, transformed version for publication

## Defaults

- Output PNG 1920x1080 with white background
- Use Inter (fallback to system-ui, sans-serif)
- Apply Catppuccin Latte accents

## Tooling workflow

- Expect an input image named `<img>-original.png` in `/artifacts`
- Extract structure into nodes, edges, groups, and annotations
- If the rendering tool is not specified, ask whether to use Mermaid, Nano Banana, or Seaborn
- Draft a render spec (YAML or Mermaid) for the cleaned layout
- Load `mermaid.md` if Mermaid is chosen
- Load `nano-banana.md` if Nano Banana is chosen
- Load `seaborn.md` if Seaborn is chosen
- Render a first pass PNG named `<img>-transformed.png` in `/artifacts`
- Compare `<img>-transformed.png` to `<img>-original.png` and iterate layout or styling to match
- Export the final PNG and confirm the filename

## Checklist

- Identify diagram type and main flow
- List elements and relationships
- Re-layout for readability
- Confirm export filename and location
