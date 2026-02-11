---
description: Convert a diagram brief into a renderable spec
---

Convert the request into a renderable spec for the visualizer agent.

Output a YAML block with:

- title
- nodes: id, label, type
- edges: from, to, label
- groups: id, label, node_ids
- annotations: text, placement
- palette: accent color choices

Keep labels short, noun-first, and consistent with the prompt.
If the diagram type is unclear, ask a single clarification question first.
