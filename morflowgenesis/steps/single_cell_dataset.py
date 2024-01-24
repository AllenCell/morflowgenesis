import hashlib
import json
import os
import re
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
from aicsimageio.writers import OmeTiffWriter
from omegaconf import ListConfig
from prefect import flow, task
from scipy.ndimage import find_objects
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.transform import rescale, resize

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    create_task_runner,
    submit,
    to_list,
)


def upload_file(
    fms_env: str,
    file_path: Path,
    intended_file_name: str,
    prov_input_file_id: str,
    prov_algorithm: str,
):
    """Upload a file located on the Isilon to FMS."""
    raise NotImplementedError


def reshape(img, z_res, xy_res, qcb_res, order=0):
    return rescale(
        img,
        (z_res / qcb_res, xy_res / qcb_res, xy_res / qcb_res),
        order=order,
        preserve_range=True,
        # multichannel=False,
    ).astype(np.uint8)


def centroid_from_slice(slicee):
    return [(s.start + s.stop) // 2 for s in slicee]


def roi_from_slice(slicee):
    return "[" + ",".join([f"{s.start},{s.stop}" for s in slicee]) + "]"


def get_renamed_image_paths(image_object, steps, rename_steps):
    assert len(rename_steps) == len(
        steps
    ), "Renaming field must be None or match 1:1 with the original names."
    return {
        rename_steps[steps.index(step_name)] + "_path": image_object.get_step(step_name).path
        for step_name in steps
    }


def get_largest_cc(im):
    im = label(im)
    try:
        largest_cc = np.argmax(np.bincount(im.flatten())[1:]) + 1
    except ValueError:
        return im
    return im == largest_cc


@task(log_prints=True)
def extract_cell(
    image_object,
    output_name,
    raw_images,
    seg_images,
    roi,
    coords,
    lab,
    raw_steps,
    seg_steps,
    is_edge,
    qcb_res,
    z_res,
    xy_res,
    upload_fms=False,
    dataset_name="morphogenesis",
    tracking_df=None,
    seg_steps_rename=None,
    raw_steps_rename=None,
):
    # prepare metadata for csv
    roi = roi_from_slice(roi)
    cell_id = hashlib.sha224(bytes(image_object.id + roi, "utf-8")).hexdigest()

    centroid = centroid_from_slice(coords)

    df = {
        "CellId": cell_id,
        "roi": roi,
        "scale_micron": str([qcb_res] * 3),
        "centroid_x": centroid[2],
        "centroid_y": centroid[1],
        "centroid_z": centroid[0],
        "label_img": lab,
        "edge_cell": is_edge,
    }
    if tracking_df is not None:
        tracking_df = tracking_df[tracking_df.label_img == lab]
        df.update(
            {
                "index_sequence": tracking_df.time_index.iloc[0],
                "track_id": tracking_df.track_id.iloc[0],
                "lineage_id": tracking_df.lineage_id.iloc[0],
                "is_outlier": tracking_df.is_outlier.iloc[0],
                "edge_cell": tracking_df.edge_cell.iloc[0],
            }
        )
    raw_steps_rename = raw_steps_rename or raw_steps
    raw_img_paths = get_renamed_image_paths(image_object, raw_steps, raw_steps_rename)
    seg_steps_rename = seg_steps_rename or seg_steps
    seg_img_paths = get_renamed_image_paths(image_object, seg_steps, seg_steps_rename)

    df.update(raw_img_paths)
    df.update(seg_img_paths)
    df = pd.DataFrame([df])

    # remove cell folder if it exists
    thiscell_path = image_object.working_dir / "single_cell_dataset" / output_name / str(cell_id)
    if os.path.isdir(thiscell_path):
        rmtree(thiscell_path)
    Path(thiscell_path).mkdir(parents=True, exist_ok=True)

    # anisotropic resize and rename dict keys
    raw_images = {
        raw_steps_rename[raw_steps.index(k)]: reshape(v, z_res, xy_res, qcb_res, order=3)
        for k, v in raw_images.items()
    }
    seg_images = {
        seg_steps_rename[seg_steps.index(k)]: reshape(v, z_res, xy_res, qcb_res, order=0)
        for k, v in seg_images.items()
    }

    # save out raw and segmentation single cell images
    name_dict = {}
    for output_type, data in zip(["raw", "seg"], [raw_images, seg_images]):
        channel_names = sorted(data.keys())
        # possible that there is no raw or no segmented image available for this cell
        if len(channel_names) == 0:
            continue
        # stack segmentation/raw images into multichannel image
        imgs = np.asarray([data[k] for k in channel_names])
        save_path = thiscell_path / f"{output_type}.tiff"
        OmeTiffWriter().save(
            uri=save_path, data=imgs, dimension_order="CZYX", channel_names=channel_names
        )

        FMS_meta = {"id": np.nan, "path": save_path}
        if upload_fms:
            crop_FMS = upload_file(
                fms_env="prod",
                file_path=save_path,
                intended_file_name=cell_id + f"_{output_type}.tiff",
                prov_input_file_id=dataset_name,
                prov_algorithm="Crop and resize",
            )

            FMS_meta["id"] = crop_FMS.id
            FMS_meta["path"] = crop_FMS.path
        df[f"crop_{output_type}_id"] = FMS_meta["id"]
        df[f"crop_{output_type}_path"] = FMS_meta["path"]
        df[f"channel_names_{output_type}"] = str(channel_names)
        name_dict[f"crop_{output_type}"] = channel_names
    df["name_dict"] = json.dumps(name_dict)
    print("cell_id", cell_id, "done")
    return df


def pad_slice(s, padding, constraints):
    # pad slice by padding subject to image size constraints
    is_edge = False
    new_slice = [slice(None, None)]
    for slice_part, c in zip(s, constraints):
        if slice_part.start == 0 or slice_part.stop >= c:
            is_edge = True
        start = max(0, slice_part.start - padding)
        stop = min(c, slice_part.stop + padding)
        new_slice.append(slice(start, stop, None))
    return tuple(new_slice), is_edge


def mask_images(
    raw_images,
    seg_images,
    raw_steps,
    seg_steps,
    lab,
    splitting_ch,
    coords,
    mask,
    keep_lcc,
    iou_thresh=None,
):
    """Turn multich image into single cell dicts."""
    # crop
    if raw_images is not None:
        raw_crop = raw_images[coords].copy()
    seg_crop = seg_images[coords].copy()
    seg_crop[splitting_ch] = seg_crop[splitting_ch] == lab
    mask_img = seg_crop[splitting_ch]

    for ch in range(len(seg_crop)):
        if ch in mask:
            seg_crop[ch] *= mask_img
        if ch in keep_lcc:
            seg_crop[ch] = get_largest_cc(seg_crop[ch] * mask_img)

    if iou_thresh is not None:
        gt = seg_crop[splitting_ch]
        for ch in range(len(seg_crop)):
            if ch == splitting_ch:
                continue
            pred = seg_crop[ch]
            intersection = np.sum(np.logical_and(gt, pred)) + 1e-8
            union = np.sum(np.logical_or(gt, pred)) + 1e-8
            iou = intersection / union
            if iou < iou_thresh:
                return None, None

    # split into dict
    return {name: raw_crop[idx] for idx, name in enumerate(raw_steps)}, {
        name: seg_crop[idx] for idx, name in enumerate(seg_steps)
    }


def load_images(image_object, splitting_step, seg_steps, raw_steps):
    """load into multichannel images."""
    available_steps = list(image_object.steps.keys())
    for i in range(len(seg_steps)):
        if "*" in seg_steps[i]:
            found_steps = [step for step in available_steps if re.search(seg_steps[i], step)]
            if found_steps is not None:
                del seg_steps[i]
                seg_steps += found_steps
            else:
                raise ValueError(
                    f"Regex search for seg_name `{seg_steps[i]}` did not find any matches. If regex search is not intended, remove `*` from seg_name"
                )
    assert splitting_step in seg_steps, "Splitting step must be included in `seg_steps`"

    seg_images = [image_object.load_step(step_name) for step_name in seg_steps]
    raw_images = [image_object.load_step(step_name) for step_name in raw_steps]
    has_raw = len(raw_images) > 0
    if has_raw:
        raw_images = [
            np.clip(im, np.percentile(im, 0.01), np.percentile(im, 99.99)) for im in raw_images
        ]
        raw_images = [
            rescale_intensity(im, out_range=np.uint8).astype(np.uint8) for im in raw_images
        ]
        raw_images = [
            resize(image, seg_images[0].shape, order=len(image.shape), preserve_range=True)
            for image in raw_images
        ]

    # some cytodl models produce models off by 1 pix due to resizing/rounding errors
    minimum_shape = np.min([im.shape for im in seg_images + raw_images], axis=0)
    minimum_shape_slice = tuple(slice(0, s) for s in minimum_shape)
    seg_images = np.stack([im[minimum_shape_slice] for im in seg_images])
    raw_images = np.stack([im[minimum_shape_slice] for im in raw_images]) if has_raw else None

    splitting_ch = seg_steps.index(splitting_step)
    return raw_images, seg_images, raw_steps, seg_steps, splitting_ch


@task(log_prints=True, tags=["benji_50"])
def process_image(
    image_object,
    splitting_step,
    padding,
    mask,
    keep_lcc,
    iou_thresh,
    output_name,
    seg_steps,
    raw_steps,
    seg_steps_rename,
    raw_steps_rename,
    xy_res,
    z_res,
    qcb_res,
    upload_fms,
    include_edge_cells,
    run_within_object,
    tracking_df=None,
):
    print(f"Processing image {image_object.id}")
    raw_images, seg_images, raw_steps, seg_steps, splitting_ch = load_images(
        image_object, splitting_step, seg_steps, raw_steps
    )
    print("images loaded for", image_object.id)
    # find objects in segmentation
    regions = find_objects(seg_images[splitting_ch].astype(int))

    # mask = True means mask all seg images, mask = list means mask only those seg images
    if mask is True:
        mask = list(range(len(seg_steps)))
    elif isinstance(mask, (list, ListConfig)):
        mask = sorted(seg_steps.index(m) for m in mask)
    else:
        mask = []

    # same for keep_lcc
    if keep_lcc is True:
        keep_lcc = list(range(len(seg_steps)))
    elif isinstance(keep_lcc, (list, ListConfig)):
        keep_lcc = sorted(seg_steps.index(m) for m in keep_lcc)
    else:
        keep_lcc = []

    results = []
    for lab, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        padded_coords, is_edge = pad_slice(coords, padding, seg_images[splitting_ch].shape)
        # do cropping serially to avoid memory blow up
        crop_raw_images, crop_seg_images = mask_images(
            raw_images,
            seg_images,
            raw_steps,
            seg_steps,
            lab,
            splitting_ch,
            padded_coords,
            mask=mask,
            keep_lcc=keep_lcc,
            iou_thresh=iou_thresh,
        )
        print("processing cell", lab, "of", image_object.id)

        if crop_raw_images is None or crop_seg_images is None:
            print(f"Skipping cell {lab} due to low IoU")
            continue
        results.append(
            submit(
                extract_cell,
                as_task=run_within_object,
                image_object=image_object,
                output_name=output_name,
                raw_images=crop_raw_images,
                seg_images=crop_seg_images,
                roi=coords,
                coords=coords,
                lab=lab,
                raw_steps=raw_steps,
                seg_steps=seg_steps,
                is_edge=is_edge,
                qcb_res=qcb_res,
                z_res=z_res,
                xy_res=xy_res,
                upload_fms=False,
                dataset_name="morphogenesis",
                tracking_df=tracking_df,
                seg_steps_rename=seg_steps_rename,
                raw_steps_rename=raw_steps_rename,
            )
        )
    return results


@flow(task_runner=create_task_runner(), log_prints=True)
def single_cell_dataset(
    image_object_paths,
    output_name,
    splitting_step,
    seg_steps,
    raw_steps=[],
    raw_steps_rename=None,
    seg_steps_rename=None,
    tracking_step=None,
    xy_res=0.108,
    z_res=0.29,
    qcb_res=0.108,
    padding=10,
    mask=True,
    keep_lcc=False,
    upload_fms=False,
    iou_thresh=None,
    include_edge_cells=True,
):

    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    # load timepoint's tracking data if available
    tracking_df = None
    if tracking_step is not None:
        tracking_df = image_objects[0].load_step(tracking_step)

    all_results = []
    for obj in image_objects:
        obj_tracking = (
            tracking_df[tracking_df.time_index == obj.metadata["T"]]
            if tracking_df is not None
            else None
        )
        all_results.append(
            submit(
                process_image,
                as_task=not run_within_object,
                image_object=obj,
                splitting_step=splitting_step,
                padding=padding,
                mask=mask,
                keep_lcc=keep_lcc,
                iou_thresh=iou_thresh,
                output_name=output_name,
                seg_steps=seg_steps,
                raw_steps=raw_steps,
                seg_steps_rename=seg_steps_rename,
                raw_steps_rename=raw_steps_rename,
                xy_res=xy_res,
                z_res=z_res,
                qcb_res=qcb_res,
                upload_fms=upload_fms,
                include_edge_cells=include_edge_cells,
                run_within_object=run_within_object,
                tracking_df=obj_tracking,
            )
        )
    for object_result, obj in zip(all_results, image_objects):
        if run_within_object:
            # parallelizing across cells within an fov
            object_result = [r.result() for r in object_result]
        else:
            # parallelizing across fovs
            object_result = object_result.result()

        if len(object_result) == 0:
            print(f"No cells found in image {obj.id}")
            continue
        df = pd.concat(object_result)

        csv_output_path = obj.working_dir / "single_cell_dataset" / output_name / f"{obj.id}.csv"

        step_output = StepOutput(
            obj.working_dir,
            "single_cell_dataset",
            output_name,
            output_type="csv",
            image_id=obj.id,
            path=csv_output_path,
        )
        step_output.save(df)

        obj.add_step_output(step_output)
        obj.save()
