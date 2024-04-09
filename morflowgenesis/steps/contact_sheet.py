from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from omegaconf import ListConfig
from skimage.exposure import rescale_intensity
from skimage.io import imsave
from skimage.segmentation import find_boundaries

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    extract_objects,
    parallelize_across_images,
)


def make_rgb(img, contour):
    """Creates RGB overlay of image and up to 3 contours in C, M, Y."""
    # normalize image intensities
    img = np.clip(img, np.percentile(img, 0.1), np.percentile(img, 99.9))
    img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
    rgb = np.stack([img] * 3, axis=-1).astype(float)
    # colors correspond to Cyan, Magenta, Yellow respectively
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for ch in range(contour.shape[0]):
        rgb[contour[ch] > 0] = colors[ch]
    return rgb.astype(np.uint8)


def project(raw, seg):
    """Create top, and both side projections of seg overlaid on raw."""
    assert len(seg.shape) == 4
    if np.all(seg == 0):
        mid_z, mid_y, mid_x = 0, 0, 0
    else:
        _, z, y, x = np.where(seg > 0)
        mid_z, mid_y, mid_x = int(np.median(z)), int(np.median(y)), int(np.median(x))

    # raw is zyx, seg is czyx
    z_project = make_rgb(raw[mid_z], seg[:, mid_z])
    y_project = make_rgb(raw[:, mid_y], seg[:, :, mid_y])
    x_project = make_rgb(raw[:, :, mid_x], seg[:, :, :, mid_x])
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

    return out.astype(np.uint8)


def project_cell(row, raw_name, seg_names):
    """Project a cell from raw and seg images."""
    print(f"{row['CellId'].iloc[0]}: starting")

    raw = AICSImage(row["crop_raw_path"].iloc[0])
    raw = raw.get_image_data("ZYX", C=raw.channel_names.index(raw_name))

    seg = AICSImage(row["crop_seg_path"].iloc[0])
    seg_channels = seg.channel_names
    if seg_names is not None:
        seg_channels = [seg.channel_names.index(n) for n in seg_names]
    seg = seg.get_image_data("CZYX", C=seg_channels).astype(np.uint8)

    seg = np.stack([find_boundaries(seg[ch], mode="inner") for ch in range(seg.shape[0])])

    projection = project(raw, seg)
    print(f"{row['CellId'].iloc[0]}: projected")

    return projection, row["CellId"].iloc[0]


def assemble_contact_sheet(results, x_bins, y_bins, x_feature, y_feature, title="Contact Sheet"):
    fig, ax = plt.subplots(len(x_bins), len(y_bins), figsize=(4 * len(x_bins), 4 * len(y_bins)))
    fig.suptitle(title)
    fig.supxlabel(x_feature)
    fig.supylabel(y_feature)
    for x_idx in range(len(x_bins)):
        for y_idx in range(len(y_bins)):
            if len(results) == 0:
                break
            img, cellid = results.pop(0)
            if img is not None:
                ax[x_idx, y_idx].imshow(img)
                ax[x_idx, y_idx].set_aspect("equal")
                ax[x_idx, y_idx].set_title(cellid, fontdict={"fontsize": 6})
                ax[x_idx, y_idx].axis("off")
    return fig


def find_cells_to_plot(n_bins, feature_df, x_feature, y_feature, cell_df):
    """Select random cells from each bin of the binned feature space."""
    quantile_boundaries = [i / n_bins for i in range(n_bins + 1)]
    # Use qcut to bin the DataFrame by percentiles across both features
    x_binned = pd.qcut(feature_df[x_feature], q=quantile_boundaries, duplicates="drop")
    y_binned = pd.qcut(feature_df[y_feature], q=quantile_boundaries, duplicates="drop")
    cells = []
    for x_bin in x_binned.unique():
        for y_bin in y_binned.unique():
            bin = np.logical_and(
                x_binned == x_bin,
                x_binned == y_bin,
            )
            if len(bin) > 0:
                cell_id = np.random.choice(bin.index.get_level_values("CellId").values)
                cell = cell_df[cell_df["CellId"] == cell_id]
                cells.append(cell)
            else:
                cells.append(None)
    return cells, x_binned, y_binned


def generate_fov_contact_sheet(image_object, output_name, raw_name, seg_step):
    """Generate a contact sheet of all cells in a fov."""
    raw = image_object.load_step(raw_name)
    seg = image_object.load_step(seg_step)
    min_im_shape = np.min([raw.shape, seg.shape], axis=0)
    # crop to same shape
    raw = raw[: min_im_shape[0], : min_im_shape[1], : min_im_shape[2]]
    seg = seg[: min_im_shape[0], : min_im_shape[1], : min_im_shape[2]]

    objects = extract_objects(seg, padding=10)
    cells = []
    for val, coords, _ in objects:
        seg_crop = find_boundaries(seg[coords] == val, mode="inner")[None]
        cells.append((project(raw[coords], seg_crop), val))

    title = f"Contact Sheet {seg_step}"
    n_bins = int(np.ceil(np.sqrt(len(cells))))
    contact_sheet = assemble_contact_sheet(
        cells, range(n_bins), range(n_bins), "", "", title=title
    )
    output = StepOutput(
        image_object.working_dir,
        "segmentation_contact_sheet_fov",
        output_name,
        output_type="image",
        image_id=image_object.id,
    )
    contact_sheet.savefig(output.path, dpi=300)
    image_object.add_step_output(output)
    image_object.save()


def segmentation_contact_sheet(
    image_objects: List[ImageObject],
    tags: List[str],
    output_name: str,
    single_cell_dataset_step: str,
    feature_step: str,
    segmentation_name: str,
    x_feature: str,
    y_feature: str,
    raw_name: str,
    n_bins: int = 10,
    seg_names: List[str] = None,
):
    """
    Create a contact sheet of random cells from the single cell dataset binned by two features.
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to create contact sheet from
    tags : List[str]
        Tags corresponding to concurrency-limits for parallel processing
    output_name : str
        Name of output
    single_cell_dataset_step : str
        Step name of single cell dataset
    feature_step : str
        Step name of calculated features
    segmentation_name : str
        Name of the segmentation to use for creating feature bins
    x_feature : str
        Name of x feature to bin by
    y_feature : str
        Name of y feature to bin by
    raw_name : str
        Name of raw image to use for projection
    n_bins : int
        Number of bins to use for each feature
    seg_names : List[str]
        List of segmentation names to use for creating contact sheet
    """
    if isinstance(seg_names, (list, ListConfig)) and len(seg_names) > 3:
        raise ValueError("Only three segmentation names can be used to create a contact sheet")
    cell_df = pd.concat(
        [image_object.load_step(single_cell_dataset_step) for image_object in image_objects]
    )
    feature_df = pd.concat(
        [image_object.load_step(feature_step) for image_object in image_objects]
    )
    feature_df = feature_df.xs(segmentation_name, level="Name")

    # project random cells from each quantile bin
    plotting_cells, x_binned, y_binned = find_cells_to_plot(
        n_bins, feature_df, x_feature, y_feature, cell_df
    )
    _, results = parallelize_across_images(
        plotting_cells,
        project_cell,
        tags,
        data_name="row",
        raw_name=raw_name,
        seg_names=seg_names,
    )
    results = [r if r is not None else (None, None) for r in results]

    # assemble contact sheet
    colors = ["Cyan", "Magenta", "Yellow"]
    title = "Contact Sheet: " + ", ".join(
        [f"{col}: {name}" for col, name in zip(colors, seg_names)]
    )
    contact_sheet = assemble_contact_sheet(
        results, x_binned.unique(), y_binned.unique(), x_feature, y_feature, title=title
    )

    output = StepOutput(
        image_objects[0].working_dir,
        "segmentation_contact_sheet",
        output_name,
        output_type="image",
        image_id=f"contact_sheet_{x_feature}_vs_{y_feature}",
    )
    contact_sheet.savefig(output.path, dpi=300)
    for image_object in image_objects:
        image_object.add_step_output(output)
        image_object.save()


def segmentation_contact_sheet_all(
    image_objects: List[ImageObject],
    output_name: str,
    raw_name: str,
    seg_step: str,
    tags: List[str],
):
    parallelize_across_images(
        image_objects,
        generate_fov_contact_sheet,
        tags,
        data_name="image_object",
        output_name=output_name,
        raw_name=raw_name,
        seg_step=seg_step,
    )


def generate_single_cell_overlays(
    image_object, single_cell_dataset_step, raw_name, seg_names, output_name
):
    """Generate a contact sheet of all cells in a fov."""
    single_cell_dataset = image_object.load_step(single_cell_dataset_step)
    for cid in single_cell_dataset.CellId.unique():
        projection, cid = project_cell(
            single_cell_dataset[single_cell_dataset.CellId == cid], raw_name, seg_names
        )
        imsave(
            f"{image_object.working_dir}/segmentation_contact_sheet/{output_name}/{cid}.png",
            projection,
        )


def segmentation_contact_sheet_cell(
    image_objects: List[ImageObject],
    output_name: str,
    single_cell_dataset_step: str,
    raw_name: str,
    seg_names: List[str],
    tags: List[str],
):
    (image_objects[0].working_dir / "segmentation_contact_sheet" / output_name).mkdir(
        exist_ok=True, parents=True
    )

    seg_colors = ["Cyan", "Magenta", "Yellow"]
    seg_key = dict(zip(seg_names, seg_colors))

    with open(
        image_objects[0].working_dir / "segmentation_contact_sheet" / output_name / "key.txt", "w"
    ) as f:
        for seg_name, color in seg_key.items():
            f.write(f"{seg_name}: {color}\n")

    parallelize_across_images(
        image_objects,
        generate_single_cell_overlays,
        tags,
        data_name="image_object",
        output_name=output_name,
        raw_name=raw_name,
        seg_names=seg_names,
        single_cell_dataset_step=single_cell_dataset_step,
    )
