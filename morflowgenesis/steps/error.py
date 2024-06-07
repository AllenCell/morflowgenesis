from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as py
import torch
from bioio import BioImage
from monai.metrics import compute_hausdorff_distance

from morflowgenesis.utils import ImageObject, parallelize_across_images


def get_surface_distance(
    image_object: ImageObject,
    single_cell_dataset_step: str,
    label_name: str,
    comparison_names: List[str],
    percentile: int = 50,
):
    dataset = image_object.load_step(single_cell_dataset_step)
    distances = {"CellId": [], "Name": [], "HausdorffDistance": []}
    for row in dataset.itertuples():
        crop = BioImage(row.crop_seg_path)
        channel_names = crop.channel_names
        crop = crop.data.squeeze()
        gt = crop[channel_names.index(label_name)]
        comparison = crop[[channel_names.index(cn) for cn in comparison_names]]
        comparison = torch.from_numpy(comparison).unsqueeze(1)
        gt = torch.from_numpy(gt).unsqueeze(0).repeat(comparison.shape[0], 1, 1, 1, 1)

        dist = compute_hausdorff_distance(
            comparison,
            gt,
            include_background=True,
            distance_metric="euclidean",
            percentile=percentile,
        )
        for i, d in enumerate(dist):
            distances["CellId"].append(row.CellId)
            distances["Name"].append(comparison_names[i])
            distances["HausdorffDistance"].append(d.item())
    return pd.DataFrame(distances)


def plot(results: pd.DataFrame, save_path: Path):
    fig = go.Figure()

    # group by name and plot quantile for hausdorff distance
    for group in results.Name.unique():
        quantiles = results[results.Name == group]["HausdorffDistance"].quantile(
            np.arange(0, 1, 0.01)
        )
        fig.add_trace(go.Scatter(x=quantiles.index, y=quantiles.values, mode="lines", name=group))
    fig.update_xaxes(title_text="Percentile")
    fig.update_yaxes(title_text="Hausdorff Surface Distance (Pixels)")
    fig.update_layout(height=600, width=600)
    py.write_html(fig, save_path)


def cell_error_metric(
    image_objects: List[ImageObject],
    output_name: str,
    single_cell_dataset_step: str,
    label_name: str,
    comparison_names: List[str],
    percentile: int = 50,
    tags: List[str] = [],
):

    if label_name in comparison_names:
        raise ValueError("Calculate_name cannot be in apply_names")

    save_path = image_objects[0].working_dir / "cell_error_metric" / output_name
    save_path.mkdir(parents=True, exist_ok=True)

    _, results = parallelize_across_images(
        image_objects,
        get_surface_distance,
        tags,
        single_cell_dataset_step=single_cell_dataset_step,
        label_name=label_name,
        comparison_names=comparison_names,
        percentile=percentile,
    )
    results = pd.concat(results)
    results.to_csv(save_path / f"hausdorff_{percentile}.csv", index=False)
    plot(results, save_path / f"hausdorff_{percentile}.html")
