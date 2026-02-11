---
description: Create a new technical diagram from a description
agent: diagrammer
---
Create a new diagram from the user description.

Requirements:
- Output a PNG at 1920x1080
- White background (non-transparent)
- Consistent font: Inter, system-ui, sans-serif
- Use Catppuccin Latte accents (see the diagram-style-catppuccin skill)
- Save the PNG in the current working directory

Process:
1. Ask any clarification questions needed to define nodes, edges, and groups.
2. Draft a diagram spec with nodes, edges, and labels.
3. Render the final PNG and report the filename.
