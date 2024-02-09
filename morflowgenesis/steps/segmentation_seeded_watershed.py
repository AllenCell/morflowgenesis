from typing import List, Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from skimage.measure import label
from skimage.segmentation import watershed

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    extract_objects,
    get_largest_cc,
    parallelize_across_images,
)


def generate_bg_seed(seg, lab):
    """returns background seeds where 2 is border and 3 is other objects."""
    # other objects are good seeds
    bg = np.logical_and(seg > 0, seg != lab)
    bg = binary_erosion(bg, iterations=5).astype(int)

    # boundaries are good seeds due to padding. we have to be careful because the object could
    # be touching the edge of the fov
    border_mask = np.ones_like(seg)
    border_mask[1:-1, 1:-1, 1:-1] = 0
    border_mask[binary_dilation(seg == lab, iterations=5)] = 0

    combined_mask = np.logical_or(border_mask, bg).astype(int) * 2
    return combined_mask


def watershed_cell(raw, seg, lab, mode, is_edge, erosion=5):
    if mode == "centroid":
        seed = np.zeros_like(seg)
        centroids = np.asarray(np.where(seg == lab)).mean(axis=1).astype(int)
        seed[centroids[0], centroids[1], centroids[2]] = 1
        seed = binary_dilation(seed, iterations=3).astype(int)
        # remove seed outside of original object
        seed[~binary_erosion(seg == lab, iterations=3)] = 0

    elif mode == "erosion" and erosion is not None:
        seed = binary_erosion(seg == lab, iterations=erosion)
        if np.max(seed) == 0:
            return np.zeros_like(seg)
        seed = get_largest_cc(seed).astype(int)
    else:
        raise ValueError(f"Unknown mode {mode}, valid options are centroid or erosion")

    bg_seed = generate_bg_seed(seg, lab)
    seed += bg_seed

    raw = np.clip(raw, np.percentile(raw, 1), np.percentile(raw, 99))
    raw = gaussian_filter(raw, sigma=[0, 1, 1], truncate=3)
    seg = watershed(raw, seed, watershed_line=True) != 2

    # remove non-target object segmentations and failed segmentations
    border_mask = np.ones_like(seg, dtype=bool)
    border_mask[1:-1, 1:-1, 1:-1] = False
    if (not is_edge and np.sum(seg[border_mask]) > 1000) or (is_edge and np.mean(seg) > 0.5):
        return np.zeros_like(seg)
    return seg


def merge_instance_segs(segs, coords, img):
    lab = np.uint16(1)
    count_map = np.zeros_like(img, dtype=np.uint8)
    for c, s in zip(coords, segs):
        img[c] += s.astype(np.uint16) * lab
        if s.max() > 0:
            count_map[c][s] += 1
        lab += 1
    # remove pixels that were assigned to multiple objects
    img[count_map > 1] = 0
    return img


def watershed_fov(
    image_object,
    output_name,
    raw_input_step,
    seg_input_step,
    padding,
    include_edge,
    mode,
    erosion,
    min_seed_size,
):
    raw = image_object.load_step(raw_input_step)
    seg = image_object.load_step(seg_input_step)

    min_im_shape = np.min([raw.shape, seg.shape], axis=0)
    raw = raw[: min_im_shape[0], : min_im_shape[1], : min_im_shape[2]]
    seg = seg[: min_im_shape[0], : min_im_shape[1], : min_im_shape[2]]

    seg = label(seg)

    labs, all_coords, edges = extract_objects(seg, padding=padding, return_zip=False)
    results = []
    for lab, coords, is_edge in zip(labs, all_coords, edges):
        if not include_edge and is_edge:
            continue
        crop_raw, crop_seg = raw[coords], seg[coords]
        # skip too small seeds not touching border
        if not is_edge and np.sum(crop_seg == lab) < min_seed_size:
            continue
        results.append(
            watershed_cell(
                raw=crop_raw,
                seg=crop_seg,
                lab=lab,
                mode=mode,
                is_edge=is_edge,
                erosion=erosion,
            )
        )
        print(lab, "done")
    seg = merge_instance_segs(results, all_coords, np.zeros(raw.shape).astype(np.uint16))
    output = StepOutput(
        image_object.working_dir,
        "run_watershed",
        output_name,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(seg)
    image_object.add_step_output(output)
    image_object.save()


def run_watershed(
    image_objects: List[ImageObject],
    tags: List[str],
    output_name: str,
    raw_input_step: str,
    seg_input_step: str,
    mode: Optional[str] = "centroid",
    erosion: Optional[int] = None,
    min_seed_size: Optional[int] = 1000,
    include_edge: Optional[bool] = True,
    padding: Optional[int] = 10,
):
    """
    Apply seeded watershed across objects in an image
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to run seeded watershed on
    tags : List[str]
        Tags corresponding to concurrency-limits for parallel processing
    output_name : str
        Name of output. The input step name will be appended to this name in the format `output_name/input_step`
    raw_input_step : str
        Step names of input raw images
    seg_input_step : str
        Step names of input segmentation images
    mode : str, optional
        Method to use for seeding, by default "centroid"
    erosion : int, optional
        Number of iterations for erosion if mode is "erosion", by default None
    min_seed_size : int, optional
        Minimum number of pixels for a seed, by default 1000
    include_edge : bool, optional
        Whether to include objects on the edge of the fov, by default True
    padding : int, optional
        Padding to add to the fov, by default 10
    """
    parallelize_across_images(
        image_objects,
        watershed_fov,
        tags,
        output_name=output_name,
        raw_input_step=raw_input_step,
        seg_input_step=seg_input_step,
        mode=mode,
        erosion=erosion,
        min_seed_size=min_seed_size,
        include_edge=include_edge,
        padding=padding,
    )
