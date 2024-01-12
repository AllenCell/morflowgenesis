import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from omegaconf import ListConfig
from prefect import flow, task
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from scipy.ndimage import find_objects

from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner, submit


def make_rgb(img, contour):  # this function returns an RGB image
    img= np.clip(img, np.percentile(img, 0.1), np.percentile(img, 99.9))
    img= rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
    rgb = np.stack([img] * 3, axis=-1).astype(float)
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for ch in range(contour.shape[0]):
        rgb[contour[ch] > 0] = colors[ch]
    return rgb.astype(np.uint8)

def project(raw, seg):
    assert len(seg.shape) == 4
    if np.all(seg == 0):
        mid_z, mid_y, mid_x = 0, 0, 0
    else:
        _, z, y, x = np.where(seg > 0)
        mid_z, mid_y, mid_x = int(np.median(z)), int(np.median(y)), int(np.median(x))

    # raw is zyx, seg is czyx
    z_project = make_rgb(raw[mid_z], seg[:,mid_z]) #overlay[mid_z]
    y_project = make_rgb(raw[:, mid_y], seg[:, :, mid_y]) #overlay[:, mid_y]
    x_project = make_rgb(raw[:, :, mid_x], seg[:, :, :, mid_x]) #overlay[:, :, mid_x]
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

def pad_coords(s, padding, constraints):
    # pad slice by padding subject to image size constraints
    new_slice = []
    for slice_part, c in zip(s, constraints):
        start = max(0, slice_part.start - padding)
        stop = min(c, slice_part.stop + padding)
        new_slice.append(slice(start, stop, None))
    return tuple(new_slice)

@task
def project_cell(row, raw_name, seg_names):
    raw = AICSImage(row["crop_raw_path"].iloc[0])
    raw = raw.get_image_dask_data("ZYX", C=raw.channel_names.index(raw_name)).compute()

    seg = AICSImage(row["crop_seg_path"].iloc[0])
    seg_channels = seg.channel_names
    if seg_names is not None:
        seg_channels = [seg.channel_names.index(n) for n in seg_names]
    seg = seg.get_image_dask_data("CZYX", C=seg_channels).compute().astype(np.uint8)

    seg = np.stack([find_boundaries(seg[ch], mode="inner") for ch in range(seg.shape[0])])

    projection = project(raw, seg)
    return projection, row['CellId'].iloc[0]


def project_fov(image_object, raw_name, seg_step):
    raw = image_object.load_step(raw_name)
    seg = image_object.load_step(seg_step)
    regions = find_objects(seg)
    cells = []
    for val, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        coords = pad_coords(coords, 10, raw.shape)
        raw_crop = raw[coords]
        seg_crop = find_boundaries(seg[coords] == val,mode="inner")[None]
        cells.append((project(raw_crop, seg_crop), val))
    return cells


def assemble_contact_sheet(results, x_bins, y_bins, x_feature, y_feature, title="Contact Sheet"):
    fig, ax = plt.subplots(len(x_bins), len(y_bins), figsize=(4 * len(x_bins), 4 * len(y_bins)))
    fig.suptitle(title)
    fig.supxlabel(x_feature)
    fig.supylabel(y_feature)
    for x_idx, x_bin in enumerate(x_bins):
        for y_idx, y_bin in enumerate(y_bins):
            if len(results)==0:
                break
            img, cellid = results.pop(0)
            if img is not None:
                ax[x_idx, y_idx].imshow(img)
                ax[x_idx, y_idx].set_aspect("equal")
                ax[x_idx, y_idx].set_title(cellid, fontdict={"fontsize": 6})
                ax[x_idx, y_idx].axis("off")
    return fig


@flow(task_runner=create_task_runner(), log_prints=True)
def segmentation_contact_sheet(
    image_object_paths,
    output_name,
    single_cell_dataset_step,
    feature_step,
    segmentation_name,
    x_feature,
    y_feature,
    raw_name,
    n_bins=10,
    seg_names=None,
):
    if isinstance(seg_names, (list, ListConfig)) and len(seg_names) > 3:
        raise ValueError("Only three segmentation names can be used to create a contact sheet")
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]

    cell_df = pd.concat(
        [image_object.load_step(single_cell_dataset_step) for image_object in image_objects]
    )
    feature_df = pd.concat(
        [image_object.load_step(feature_step) for image_object in image_objects]
    )

    feature_df = feature_df.xs(segmentation_name, level="Name")

    quantile_boundaries = [i / n_bins for i in range(n_bins + 1)]

    # Use qcut to bin the DataFrame by percentiles across both features
    x_binned = pd.qcut(feature_df[x_feature], q=quantile_boundaries, duplicates="drop")
    y_binned = pd.qcut(feature_df[y_feature], q=quantile_boundaries, duplicates="drop")
    results = []
    for x_bin in x_binned.unique():
        for y_bin in y_binned.unique():
            bin = np.logical_and(
                x_binned == x_bin,
                x_binned == y_bin,
            )
            if len(bin) > 0:
                cell_id = np.random.choice(bin.index.get_level_values("CellId").values)
                cell_features = cell_df[cell_df["CellId"] == cell_id]
                results.append(project_cell.submit(cell_features, raw_name, seg_names))
            else:
                results.append(None)
    results = [r.result() if r is not None else (None, None) for r in results]
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

@flow(task_runner=create_task_runner(), log_prints=True)
def segmentation_contact_sheet_all(
    image_object_paths,
    output_name,
    raw_name,
    seg_step,
):
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]

    colors = ["Cyan", "Magenta", "Yellow"]

    for image_object in image_objects:
        cells = project_fov(image_object, raw_name, seg_step)
        title = "Contact Sheet: " + ", ".join(
            [f"{col}: {name}" for col, name in zip(colors, seg_step)]
        )
        n_bins = int(np.ceil(np.sqrt(len(cells))))
        contact_sheet = assemble_contact_sheet(
            cells, range(n_bins), range(n_bins), '', '', title=title
        )
        output = StepOutput(
            image_objects[0].working_dir,
            "segmentation_contact_sheet",
            output_name,
            output_type="image",
            image_id=image_object.id,
        )
        contact_sheet.savefig(output.path, dpi=300)
        for image_object in image_objects:
            image_object.add_step_output(output)
            image_object.save()
