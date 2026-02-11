## Seaborn rendering

- Create a local venv with `uv venv` and install `seaborn`, `matplotlib`, `pandas`, `numpy`
- Use the plotting entrypoint at `scripts/plot.py` without reading its source
- Provide inputs using the schema below and capture the PNG path from stdout
- Run with `uv run scripts/plot.py` and pass JSON via stdin
- Use `sns.set_theme(style="whitegrid")` and size to 1920x1080
- For 3D mesh-style plots, use Matplotlib's `mplot3d` toolkit
- Save a PNG with a white background and confirm the filename

## Input schema (JSON)

- chart_type: line | scatter | bar | heatmap | mesh
- data_path: path to CSV or Parquet
- x: column name
- y: column name
- z: column name (mesh only)
- value: column name (heatmap only)
- hue: column name (optional)
- title: string
- output_path: optional destination PNG path
- theme: default "whitegrid"
- size: [width, height] in pixels

## Output schema (stdout JSON)

- status: ok | error
- output_path: PNG path (temp path if output_path omitted)
- message: optional error details
