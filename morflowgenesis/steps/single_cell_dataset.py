import gc
import hashlib
import json
import re
from shutil import rmtree
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tqdm
from aicsimageio.writers import OmeTiffWriter
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from skimage.transform import rescale

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    extract_objects,
    get_largest_cc,
    parallelize_across_images,
)


def centroid_from_slice(slicee):
    return [(s.start + s.stop) // 2 for s in slicee]


def roi_from_slice(slicee):
    return "[" + ",".join([f"{s.start},{s.stop}" for s in slicee]) + "]"


def get_renamed_image_paths(
    image_object, raw_steps, raw_rename_steps, seg_steps, seg_rename_steps
):
    img_paths = {}
    for step_type, steps, rename_steps in (
        ("raw", raw_steps, raw_rename_steps),
        ("seg", seg_steps, seg_rename_steps),
    ):
        if steps is None or len(steps) == 0 or rename_steps is None or len(rename_steps) == 0:
            continue
        assert len(rename_steps) == len(
            steps
        ), "Renaming field must be None or match 1:1 with the original names."
        img_paths[step_type] = {
            rename_steps[steps.index(step_name)] + "_path": image_object.get_step(step_name).path
            for step_name in steps
        }
    return img_paths


def append_dict(features_dict, new_dict):
    for k, v in new_dict.items():
        try:
            features_dict[k].append(v)
        except KeyError:
            features_dict[k] = [v]
    return features_dict


def calculate_features(data, features, cellid):
    """
    features: dict of channel_name to apply to : [function1, function2, ...]
    """
    multi_index = []
    features_dict = {}
    for k, v in data.items():
        for feat_fn in features.get(k, []):
            features_dict = append_dict(features_dict, feat_fn(v))
        multi_index.append((cellid, k))

    features_dict = pd.DataFrame(
        features_dict,
        index=pd.MultiIndex.from_tuples(multi_index, names=["CellId", "Name"]),
    )
    if features_dict.shape[0] == 0:
        return None
    return features_dict


def prepare_metadata(image_object, roi, lab, is_edge, img_paths, out_res):
    centroid = centroid_from_slice(roi)
    roi = roi_from_slice(roi)
    cell_id = hashlib.sha224(bytes(f"{image_object.id}{roi}", "utf-8")).hexdigest()

    df = {
        "CellId": cell_id,
        "roi": roi,
        "scale_micron": out_res,
        "centroid_x": centroid[2],
        "centroid_y": centroid[1],
        "centroid_z": centroid[0],
        "label_img": lab,
        "edge_cell": is_edge,
    }
    for k, v in img_paths.items():
        df.update(v)
    return df


def setup_directories(image_object, output_name, cell_id):
    # remove cell folder if it exists
    thiscell_path = image_object.working_dir / "single_cell_dataset" / output_name / str(cell_id)
    if thiscell_path.is_dir():
        rmtree(thiscell_path)
    thiscell_path.mkdir(parents=True, exist_ok=True)
    return thiscell_path


def process_cells(crops, image_object, output_name, img_paths, out_res, features, save):
    while crops:
        crop = crops[0]
        if isinstance(crop, dict):
            result = process_cell(
                image_object, output_name, crops[0], img_paths, out_res, features, save
            )
            yield result
        del crops[0]


def process_cell(
    image_object: ImageObject,
    output_name: str,
    crop: Dict,
    img_paths: str,
    out_res: Dict[str, Union[float, np.ndarray]],
    features: Dict[str, Callable] = {},
    save: bool = True,
):
    print("processing cell", crop["lab"], "of", image_object.id)

    cell_meta = prepare_metadata(
        image_object,
        crop["roi"],
        crop["lab"],
        crop["is_edge"],
        img_paths,
        out_res,
    )
    if save:
        save_dir = setup_directories(image_object, output_name, cell_meta["CellId"])

    # save out raw and segmentation single cell images
    cell_features = []
    name_dict = {}
    for output_type in img_paths.keys():
        data = crop[output_type]
        cell_features.append(calculate_features(data, features, cell_meta["CellId"]))
        if save:
            # stack images into multichannel image
            channel_names = list(data.keys())
            imgs = np.stack(list(data.values()))
            save_path = save_dir / f"{output_type}.tiff"
            OmeTiffWriter().save(
                uri=save_path, data=imgs, dimension_order="CZYX", channel_names=channel_names
            )
            cell_meta.update(
                {
                    f"crop_{output_type}_id": np.nan,
                    f"crop_{output_type}_path": save_path,
                    f"channel_names_{output_type}": str(channel_names),
                }
            )
            name_dict[f"crop_{output_type}"] = channel_names
    cell_meta["name_dict"] = json.dumps(name_dict)
    return pd.DataFrame([cell_meta]), pd.concat(cell_features)


def _calc_iou(im1, im2):
    minimum_shape = np.minimum(im1.shape[-3:], im2.shape[-3:])
    im1 = im1[: minimum_shape[0], : minimum_shape[1], : minimum_shape[2]]
    im2 = im2[: minimum_shape[0], : minimum_shape[1], : minimum_shape[2]]

    intersection = np.sum(np.logical_and(im1, im2)) + 1e-8
    union = np.sum(np.logical_or(im1, im2)) + 1e-8
    return intersection / union


def reshape(img, input_res, out_res, order=0):
    return rescale(
        img,
        np.array(input_res) / np.array(out_res),
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    )


def multi_res_crop(img, coords, coords_res, res):
    ratio = np.array(coords_res) / np.array(res)
    new_coords = tuple(
        slice(int(c.start * ratio[i]), int(c.stop * ratio[i])) for i, c in enumerate(coords)
    )
    return img[new_coords].copy()


def mask_images(
    raw_images,
    seg_images,
    lab,
    splitting_step,
    coords,
    mask,
    keep_lcc,
    is_edge,
    in_res,
    out_res,
    iou_thresh=None,
):
    raw_images = {
        name: multi_res_crop(raw_images[name], coords, in_res[splitting_step], in_res[name])
        for name in raw_images
    }
    raw_images = {k: reshape(v, in_res[k], out_res[k], order=0) for k, v in raw_images.items()}
    seg_images = {
        name: multi_res_crop(seg_images[name], coords, in_res[splitting_step], in_res[name])
        for name in seg_images
    }
    seg_images = {k: reshape(v, in_res[k], out_res[k], order=0) for k, v in seg_images.items()}
    min_shape = np.min(
        [img.shape for img in list(seg_images.values()) + list(raw_images.values())], axis=0
    )
    print("Cropping to", min_shape)
    raw_images = {
        k: img[: min_shape[0], : min_shape[1], : min_shape[2]] for k, img in raw_images.items()
    }
    seg_images = {
        k: img[: min_shape[0], : min_shape[1], : min_shape[2]] for k, img in seg_images.items()
    }

    seg_images[splitting_step] = (seg_images[splitting_step] == lab).astype(np.uint8)
    mask_img = seg_images[splitting_step]

    for name, img in seg_images.items():
        if iou_thresh is not None:
            if name == splitting_step:
                continue
            if _calc_iou(mask_img, img) < iou_thresh:
                return None, None
        if name in mask:
            seg_images[name] *= mask_img
        if name in keep_lcc:
            seg_images[name] = get_largest_cc(img, mask=mask_img)

    return {
        "raw": raw_images,
        "seg": seg_images,
        "lab": lab,
        "roi": coords,
        "is_edge": is_edge,
    }


def _rename(dict, new_names):
    """if rename is None, use original dictnames if rename is list, use new names iff they have the
    same length as original names."""
    if new_names is None:
        return dict
    assert len(new_names) == len(
        dict
    ), "Renaming field must be None or match 1:1 with the original names."
    old_names = list(dict.keys())
    return {new_names[i]: dict[name] for i, name in enumerate(old_names)}


def load_images(image_object, seg_steps, raw_steps, seg_steps_rename, raw_steps_rename):
    """load, and rename images."""
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
    seg_images = {
        step_name: image_object.load_step(step_name)
        for step_name in tqdm.tqdm(seg_steps, desc="Loading Segmentation Images")
    }
    seg_images = _rename(seg_images, seg_steps_rename)

    raw_images = (
        {
            step_name: image_object.load_step(step_name)
            for step_name in tqdm.tqdm(raw_steps, desc="Loading Raw Images")
        }
        if raw_steps
        else {}
    )

    raw_images = _rename(raw_images, raw_steps_rename)

    for k, v in seg_images.items():
        print(k, v.shape)
    for k, v in raw_images.items():
        print(k, v.shape)
    return raw_images, seg_images, raw_steps, seg_steps


def get_apply_channels(filter_type, channels):
    # filter_type = True means apply to all channels, filter_type is list means mask channels in list
    if filter_type is True:
        return channels
    elif isinstance(filter_type, (list, ListConfig)):
        return [c for c in channels if c in filter_type]
    return []


def generate_dict_with_default(data, keys):
    if isinstance(data, (dict, DictConfig)):
        assert all(key in data.keys() for key in keys), f"Expected keys {data.keys()}, got {keys}"
        return data
    return {k: data for k in keys}


def extract_cells_from_fov(
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
    input_res,
    out_res,
    include_edge_cells,
    features={},
    save: bool = True,
):
    print(f"Processing image {image_object.id}")
    raw_images, seg_images, raw_steps, seg_steps = load_images(
        image_object, seg_steps, raw_steps, seg_steps_rename, raw_steps_rename
    )
    print("images loaded for", image_object.id)

    # instantiate feature calculation classes
    features = generate_dict_with_default(features, seg_images.keys())
    features = {k: [instantiate(feat) for feat in v] for k, v in features.items()}

    # get resolution of each image
    all_keys = list(seg_images.keys()) + list(raw_images.keys())
    input_res = generate_dict_with_default(input_res, all_keys)
    out_res = generate_dict_with_default(out_res, all_keys)

    mask = get_apply_channels(mask, seg_images.keys())
    keep_lcc = get_apply_channels(keep_lcc, seg_images.keys())
    print(
        "Masking channels:",
        mask,
        "\n",
        "Keeping only largest connected component for channels:",
        keep_lcc,
    )

    img_paths = get_renamed_image_paths(
        image_object, raw_steps, list(raw_images.keys()), seg_steps, list(seg_images.keys())
    )

    # find objects in segmentation
    if seg_images[splitting_step].dtype != np.uint16:
        seg_images[splitting_step] = seg_images[splitting_step].astype(np.uint16)
    objects = extract_objects(seg_images[splitting_step], padding=padding)
    print("Object coords extracted")
    crops = [
        mask_images(
            raw_images,
            seg_images,
            lab,
            splitting_step,
            coords,
            mask,
            keep_lcc,
            is_edge,
            input_res,
            out_res,
            iou_thresh,
        )
        for lab, coords, is_edge in tqdm.tqdm(objects, desc="masking objects")
    ]

    # original images are no longer needed
    del raw_images
    del seg_images
    gc.collect()

    cell_info = list(
        process_cells(crops, image_object, output_name, img_paths, out_res, features, save)
    )

    cell_meta = pd.concat([c[0] for c in cell_info])
    cell_features = pd.concat([c[1] for c in cell_info])

    step_output = StepOutput(
        image_object.working_dir,
        "single_cell_dataset",
        output_name,
        output_type="csv",
        image_id=image_object.id,
    )
    step_output.save(cell_meta)
    image_object.add_step_output(step_output)

    if cell_features.shape[0] > 0:
        step_output = StepOutput(
            image_object.working_dir,
            "calculate_features",
            output_name,
            output_type="csv",
            image_id=image_object.id,
            index_col=["CellId", "Name"],
        )
        step_output.save(cell_features)
        image_object.add_step_output(step_output)
    image_object.save()


def single_cell_dataset(
    image_objects: List[ImageObject],
    tags: List[str],
    output_name: str,
    splitting_step: str,  # if splitting step is none, run features at fov level??
    seg_steps: List[str],
    raw_steps: Optional[List[str]] = [],
    raw_steps_rename: Optional[List[str]] = None,
    seg_steps_rename: Optional[List[str]] = None,
    input_res: Optional[Union[Dict, List[float]]] = {},
    out_res: Optional[Union[Dict, List[float], float]] = {},
    padding: Optional[Union[int, List[int]]] = 10,
    mask: Optional[Union[bool, List[str]]] = True,
    keep_lcc: Optional[Union[bool, List[str]]] = False,
    iou_thresh: Optional[float] = None,
    include_edge_cells: Optional[bool] = True,
    features: Optional[Union[Dict, List[Dict]]] = {},
    save: bool = True,
):
    """
    Create a single cell dataset from a set of images and segmentation masks.
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to create single cell dataset from
    tags : List[str]
        Tags corresponding to concurrency-limits for parallel processing
    output_name : str
        Name of output.
    splitting_step : str
        Step name of input segmentation image to use for splitting fov into single cells
    seg_steps : List[str]
        Step names of input segmentation images
    raw_steps : List[str], optional
        Step names of input raw images
    raw_steps_rename : List[str], optional
        New names for raw steps
    seg_steps_rename : List[str], optional
        New names for seg steps
    input_res: float or Dict[List[float]], optional
        Dictionary of zyx resolution for each image in seg_steps and raw_steps. If none is provided, it is assumed images are already at desired resolution, if only one list is provided, it is assumed to be the resolution for all images
    out_res : float, optional
        Dictionary of zyx resolution for each image in seg_steps and raw_steps. If none is provided, it is assumed images are already at desired resolution
    padding : int or List[int], optional
        Padding around each object. If single int, same for all axes, otherwise padding is per-axis
    mask : bool or List[str], optional
        Whether to mask segmentation images. If list, only mask those channels
    keep_lcc :
        Whether to keep only the largest connected component of each channel in the segmentation images. If list, only keep lcc for those channels
    upload_fms : bool, optional
        Whether to upload single cell images to FMS. Not implemented
    iou_thresh : float, optional
        Minimum IoU between splitting step and other objects to keep a cell
    include_edge_cells : bool, optional
        Whether to include cells on the edge of the fov
    features : Dict[str, List[Dict]], optional
        Dictionary like name : [function1, function2, ...] to apply to each channel in the single cell images OR just [function1, function2, ...] to apply to all segmentations. If None, no features are calculated
    save : bool, optional
        Whether to save single cell images to disk
    """
    parallelize_across_images(
        image_objects,
        extract_cells_from_fov,
        tags=tags,
        delay=0.5,
        output_name=output_name,
        splitting_step=splitting_step,
        padding=padding,
        mask=mask,
        keep_lcc=keep_lcc,
        iou_thresh=iou_thresh,
        seg_steps=seg_steps,
        raw_steps=raw_steps,
        seg_steps_rename=seg_steps_rename,
        raw_steps_rename=raw_steps_rename,
        input_res=input_res,
        out_res=out_res,
        include_edge_cells=include_edge_cells,
        features=features,
        save=save,
    )
