---
description: Recreate a technical diagram from an image
agent: diagrammer
---
You are given an input image of a technical diagram.
Recreate it in a transformed style suitable for an article.

Requirements:
- Output a PNG at 1920x1080
- White background (non-transparent)
- Consistent font: Inter, system-ui, sans-serif
- Use Catppuccin Latte accents (see the diagram-style-catppuccin skill)
- Prefer clarity over pixel-perfect replication
- Save the PNG in the current working directory

Process:
1. Summarize the image into nodes, edges, groups, and annotations.
2. Normalize labels (short, noun-first).
3. Lay out the diagram for clarity with balanced spacing.
4. Render the final PNG and report the filename.
