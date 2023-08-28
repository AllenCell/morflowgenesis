import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from prefect import flow, task
from skimage.segmentation import find_boundaries

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import StepOutput


def make_rgb(img, contour):  # this function returns an RGB image
    rgb = np.stack([img] * 3, axis=-1)
    rgb[contour > 0, 0] = 255  # red channel
    rgb[contour > 0, 1] = 0  # green channel
    rgb[contour > 0, 2] = 255  # blue channel
    return rgb


@task
def project_cell(row, raw_channel, seg_channel):
    raw = (
        AICSImage(row["crop_raw_path"])
        .get_image_dask_data("ZYX", C=raw_channel)
        .compute()
        .astype(np.uint8)
    )
    seg = (
        AICSImage(row["crop_seg_path"])
        .get_image_dask_data("ZYX", C=seg_channel)
        .compute()
        .astype(np.uint8)
    )
    seg = find_boundaries(seg, mode="inner")

    z, y, x = np.where(seg > 0)
    mid_z, mid_y, mid_x = int(np.median(z)), int(np.median(y)), int(np.median(x))

    overlay = make_rgb(raw, seg)

    z_project = overlay[mid_z]
    y_project = overlay[:, mid_y]
    x_project = overlay[:, :, mid_x]
    x_project = np.transpose(x_project, (1, 0, 2))

    buffer = 0
    out = np.zeros(
        (
            z_project.shape[0] + buffer + x_project.shape[0],
            z_project.shape[1] + buffer + y_project.shape[1],
            3,
        )
    )
    out[: y_project.shape[0], : y_project.shape[1]] = y_project  # top left
    out[-z_project.shape[0] :, : z_project.shape[1]] = z_project  # bottom left
    out[-x_project.shape[0] :, -x_project.shape[1] :] = x_project  # bottom right

    return out.astype(np.uint8), row["CellId"]


def assemble_contact_sheet(results, x_bins, y_bins, x_characteristic, y_characteristic):
    fig, ax = plt.subplots(len(x_bins), len(y_bins), figsize=(4 * len(x_bins), 4 * len(y_bins)))
    fig.supxlabel(x_characteristic)
    fig.supylabel(y_characteristic)

    shapes = np.asarray([x.shape for x in results])
    contact_sheet = np.zeros((len(x_bins) * np.max(shapes, 0), len(y_bins) * np.max(shapes, 1)))
    for x_idx, x_bin in enumerate(x_bins):
        for y_idx, y_bin in enumerate(y_bins):
            img = results.pop(0)
            if img is not None:
                img, cellid = img.result()
                ax[x_idx, y_idx].imshow(img)
                ax[x_idx, y_idx].set_aspect("equal")
                ax[x_idx, y_idx].set_title(cellid)

    return contact_sheet


@flow(task_runner=create_task_runner(), log_prints=True)
def run_contact_sheet(
    image_object,
    step_name,
    output_name,
    single_cell_dataset_step,
    feature_step,
    x_characteristic,
    y_characteristic,
    n_bins=10,
    raw_channel=0,
    seg_channel=0,
):
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return image_object

    cell_df = image_object.load_step(single_cell_dataset_step)
    feature_df = image_object.load_step(feature_step)
    quantile_boundaries = [i / n_bins for i in range(n_bins + 1)]

    # Use qcut to bin the DataFrame by percentiles across both features
    feature_df[f"{x_characteristic}_bin"] = pd.qcut(
        feature_df[f"{x_characteristic}"], q=quantile_boundaries
    )
    feature_df[f"{y_characteristic}_bin"] = pd.qcut(
        feature_df[f"{y_characteristic}"], q=quantile_boundaries
    )

    results = []
    x_bins = feature_df[f"{x_characteristic}_bin"].unique()
    y_bins = feature_df[f"{y_characteristic}_bin"].unique()
    for x_bin in x_bins:
        for y_bin in y_bins:
            temp = feature_df[
                np.logical_and(
                    feature_df[f"{x_characteristic}_bin"] == x_bin,
                    feature_df[f"{y_characteristic}_bin"] == y_bin,
                )
            ]
            if len(temp) > 0:
                cell_id = temp["Cell_id"].sample(1)
                results.append(
                    project_cell.submit(
                        cell_df[cell_df["Cell_id"] == cell_id], raw_channel, seg_channel
                    )
                )
            else:
                results.append(None)

    contact_sheet = assemble_contact_sheet(
        results, x_bins, y_bins, x_characteristic, y_characteristic
    )

    output = StepOutput(
        image_object.output_dir,
        step_name,
        output_name,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(contact_sheet)
    image_object.add_step_output(output)
    image_object.save()
    return image_object
