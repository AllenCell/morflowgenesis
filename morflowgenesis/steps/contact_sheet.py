import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from prefect import flow, task
from skimage.segmentation import find_boundaries

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


def make_rgb(img, contour):  # this function returns an RGB image
    rgb = np.stack([img] * 3, axis=-1)
    for ch in range(contour.shape[0]):
        rgb[contour[ch] > 0, ch] = 255
    return rgb


@task
def project_cell(row, raw_channel, seg_channels):
    raw = (
        AICSImage(row["crop_raw_path"].iloc[0])
        .get_image_dask_data("ZYX", C=raw_channel)
        .compute()
        .astype(np.uint8)
    )
    seg = (
        AICSImage(row["crop_seg_path"].iloc[0])
        .get_image_dask_data("CZYX", C=seg_channels)
        .compute()
        .astype(np.uint8)
    )

    seg = np.stack([find_boundaries(seg[ch], mode="inner") for ch in range(seg.shape[0])])
    _, z, y, x = np.where(seg > 0)
    mid_z, mid_y, mid_x = int(np.median(z)), int(np.median(y)), int(np.median(x))

    overlay = make_rgb(raw, seg)

    z_project = overlay[mid_z]
    y_project = overlay[:, mid_y]
    x_project = overlay[:, :, mid_x]
    x_project = np.transpose(x_project, (1, 0, 2))

    # Calculate the required output dimensions
    out_height = y_project.shape[0] + z_project.shape[0]
    out_width = x_project.shape[1] + z_project.shape[1]

    # Create the output image with the calculated dimensions
    out = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # Place projections onto the output image
    out[: y_project.shape[0], : y_project.shape[1]] = y_project  # top left
    out[out_height - z_project.shape[0] :, : z_project.shape[1]] = z_project  # bottom left
    out[
        out_height - x_project.shape[0] :, out_width - x_project.shape[1] :
    ] = x_project  # bottom right

    return out.astype(np.uint8), row["CellId"]


def assemble_contact_sheet(
    results, x_bins, y_bins, x_characteristic, y_characteristic, title="Contact Sheet"
):
    fig, ax = plt.subplots(len(x_bins), len(y_bins), figsize=(4 * len(x_bins), 4 * len(y_bins)))
    fig.suptitle(title)
    fig.supxlabel(x_characteristic)
    fig.supylabel(y_characteristic)
    for x_idx, x_bin in enumerate(x_bins):
        for y_idx, y_bin in enumerate(y_bins):
            img, cellid = results.pop(0)
            if img is not None:
                ax[x_idx, y_idx].imshow(img)
                ax[x_idx, y_idx].set_aspect("equal")
                ax[x_idx, y_idx].set_title(cellid.values[0], fontdict={"fontsize": 6})
                ax[x_idx, y_idx].axis("off")
    return fig


@flow(task_runner=create_task_runner(), log_prints=True)
def segmentation_contact_sheet(
    image_object_paths,
    step_name,
    output_name,
    single_cell_dataset_step,
    feature_step,
    x_characteristic,
    y_characteristic,
    n_bins=10,
    raw_channel=0,
    seg_channels=[0],
):
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]

    cell_df = pd.concat(
        [image_object.load_step(single_cell_dataset_step) for image_object in image_objects]
    )
    feature_df = pd.concat(
        [image_object.load_step(feature_step) for image_object in image_objects]
    )

    quantile_boundaries = [i / n_bins for i in range(n_bins + 1)]

    # Use qcut to bin the DataFrame by percentiles across both features
    feature_df[f"{x_characteristic}_bin"] = pd.qcut(
        feature_df[f"{x_characteristic}"], q=quantile_boundaries, duplicates="drop"
    )
    feature_df[f"{y_characteristic}_bin"] = pd.qcut(
        feature_df[f"{y_characteristic}"], q=quantile_boundaries, duplicates="drop"
    )
    x_bins = feature_df[f"{x_characteristic}_bin"].unique()
    y_bins = feature_df[f"{y_characteristic}_bin"].unique()

    results = []
    for x_bin in x_bins:
        for y_bin in y_bins:
            temp = feature_df[
                np.logical_and(
                    feature_df[f"{x_characteristic}_bin"] == x_bin,
                    feature_df[f"{y_characteristic}_bin"] == y_bin,
                )
            ]
            if len(temp) > 0:
                cell_id = temp["CellId"].sample(1).values[0]
                results.append(
                    project_cell.submit(
                        cell_df[cell_df["CellId"] == cell_id], raw_channel, seg_channels
                    )
                )
            else:
                results.append(None)
    results = [r.result() if r is not None else (None, None) for r in results]

    channel_names = AICSImage(cell_df["crop_seg_path"].iloc[0]).channel_names
    colors = ["Red", "Green", "Blue"]
    title = "Contact Sheet: " + ", ".join(
        [f"{col}: {name}" for col, name in zip(colors, channel_names)]
    )
    contact_sheet = assemble_contact_sheet(
        results, x_bins, y_bins, x_characteristic, y_characteristic, title=title
    )

    output = StepOutput(
        image_objects[0].working_dir,
        step_name,
        output_name,
        output_type="image",
        image_id=f"contact_sheet_{x_characteristic}_vs_{y_characteristic}",
    )
    contact_sheet.savefig(output.path, dpi=300)
    for image_object in image_objects:
        image_object.add_step_output(output)
        image_object.save()


@flow(task_runner=create_task_runner(), log_prints=True)
def run_contact_sheet(
    image_object_path,
    step_name,
    output_name,
    single_cell_dataset_step,
    feature_step=None,
    x_characteristic=None,
    y_characteristic=None,
    n_bins=10,
    raw_channel=0,
    seg_channel=0,
    grouping_column=None,
):
    image_object = ImageObject.parse_file(image_object_path)

    cell_df = image_object.load_step(single_cell_dataset_step)
    feature_df = image_object.load_step(feature_step)
    quantile_boundaries = [i / n_bins for i in range(n_bins + 1)]

    # Use qcut to bin the DataFrame by percentiles across both features
    feature_df[f"{x_characteristic}_bin"] = pd.qcut(
        feature_df[f"{x_characteristic}"], q=quantile_boundaries, duplicates="drop"
    )
    feature_df[f"{y_characteristic}_bin"] = pd.qcut(
        feature_df[f"{y_characteristic}"], q=quantile_boundaries, duplicates="drop"
    )
    x_bins = feature_df[f"{x_characteristic}_bin"].unique()
    y_bins = feature_df[f"{y_characteristic}_bin"].unique()

    grouped_dfs = [cell_df]
    if grouping_column is not None:
        grouped_dfs = [
            cell_df[cell_df[grouping_column] == cat] for cat in cell_df.grouping_column.unique()
        ]

    for gdf in grouped_dfs:
        results = []
        for x_bin in x_bins:
            for y_bin in y_bins:
                temp = feature_df[
                    np.logical_and(
                        feature_df[f"{x_characteristic}_bin"] == x_bin,
                        feature_df[f"{y_characteristic}_bin"] == y_bin,
                    )
                ]
                if len(temp) > 0:
                    cell_id = temp["CellId"].sample(1).values[0]
                    results.append(
                        project_cell.submit(
                            gdf[gdf["CellId"] == cell_id], raw_channel, seg_channel
                        )
                    )
                else:
                    results.append(None)
        results = [r.result() if r is not None else (None, None) for r in results]

        contact_sheet = assemble_contact_sheet(
            results, x_bins, y_bins, x_characteristic, y_characteristic
        )

        group_name = "" if grouping_column is None else f"_{gdf[grouping_column].iloc[0]}"
        output = StepOutput(
            image_object.working_dir,
            step_name,
            output_name + group_name,
            output_type="image",
            image_id=image_object.id,
        )
        contact_sheet.savefig(output.path, dpi=300)
        image_object.add_step_output(output)
    image_object.save()
