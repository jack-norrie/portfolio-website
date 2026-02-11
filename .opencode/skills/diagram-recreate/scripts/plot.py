import json
import os
import sys
import tempfile
from typing import Optional, Sequence

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401


def load_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def ensure_output_path(output_path: Optional[str]) -> str:
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return output_path
    temp_dir = tempfile.mkdtemp(prefix="diagram-plot-")
    return os.path.join(temp_dir, "plot.png")


def build_plot(data: dict) -> str:
    chart_type = data.get("chart_type")
    data_path = data.get("data_path")
    x = data.get("x")
    y = data.get("y")
    z = data.get("z")
    hue = data.get("hue")
    value = data.get("value")
    title = data.get("title")
    theme = data.get("theme", "whitegrid")
    size = data.get("size", [1920, 1080])

    if not chart_type or not data_path:
        raise ValueError("chart_type and data_path are required")
    if chart_type != "mesh" and (not x or not y):
        raise ValueError("x and y are required for this chart type")
    if chart_type == "mesh" and (not x or not y or not z):
        raise ValueError("x, y, and z are required for mesh plots")

    width_px, height_px = size
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)

    sns.set_theme(style=theme)
    output_path = ensure_output_path(data.get("output_path"))
    dataframe = load_dataframe(data_path)

    if chart_type == "mesh":
        figure = plt.figure(figsize=figsize, dpi=dpi)
        axes = figure.add_subplot(111, projection="3d")
        axes.plot_trisurf(
            dataframe[x].to_numpy(),
            dataframe[y].to_numpy(),
            dataframe[z].to_numpy(),
            cmap="viridis",
            linewidth=0.2,
        )
        if title:
            axes.set_title(title)
    else:
        figure, axes = plt.subplots(figsize=figsize, dpi=dpi)
        if chart_type == "line":
            sns.lineplot(data=dataframe, x=x, y=y, hue=hue, ax=axes)
        elif chart_type == "scatter":
            sns.scatterplot(data=dataframe, x=x, y=y, hue=hue, ax=axes)
        elif chart_type == "bar":
            sns.barplot(data=dataframe, x=x, y=y, hue=hue, ax=axes)
        elif chart_type == "heatmap":
            if not value:
                raise ValueError("value is required for heatmap plots")
            pivoted = dataframe.pivot(index=y, columns=x, values=value)
            sns.heatmap(pivoted, ax=axes)
        else:
            raise ValueError(f"Unsupported chart_type: {chart_type}")

        if title:
            axes.set_title(title)

    figure.patch.set_facecolor("white")
    figure.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        output_path = build_plot(payload)
        response = {"status": "ok", "output_path": output_path}
        json.dump(response, sys.stdout)
        return 0
    except Exception as exc:  # pragma: no cover - error reporting
        response = {"status": "error", "output_path": "", "message": str(exc)}
        json.dump(response, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
