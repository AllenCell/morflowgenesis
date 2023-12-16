import numpy as np
# from mahotas import cwatershed as watershed
from skimage.segmentation import watershed
from prefect import flow, task
from scipy.ndimage import binary_dilation, binary_erosion, find_objects
from skimage.filters import median
from skimage.measure import label
from skimage.morphology import disk

from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner, submit


def pad_slice(s, padding, constraints):
    # pad slice by padding subject to image size constraints
    is_edge = False
    new_slice = []
    for slice_part, c in zip(s, constraints):
        if slice_part.start == 0 or slice_part.stop >= c:
            is_edge = True
        start = max(0, slice_part.start - padding)
        stop = min(c, slice_part.stop + padding)
        new_slice.append(slice(start, stop, None))
    return tuple(new_slice), is_edge


def get_largest_cc(im):
    im = label(im)
    largest_cc = np.argmax(np.bincount(im.flatten())[1:]) + 1
    return im == largest_cc


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

    combined_mask = border_mask + 2 * bg
    combined_mask[combined_mask > 0] += 1

    return combined_mask


@task
def run_watershed_task(raw, seg, lab, mode, is_edge, erosion=5, smooth=False):
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

    if smooth:
        raw = median(raw)
    seg = watershed(raw, seed)

    # dilate in xy into areas not covered by watershed on other objects
    selem = np.zeros((3, 3, 3))
    selem[1] = disk(1)
    seg = binary_dilation(seg == 1, iterations=1, structure=selem, mask=seg != 3)

    # remove non-target object segmentations and failed segmentations
    border_mask = np.ones_like(seg)
    border_mask[1:-1, 1:-1, 1:-1] = 0
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


@task
def run_object(
    image_object,
    raw_input_step,
    seg_input_step,
    padding,
    include_edge,
    mode,
    erosion,
    smooth,
    min_seed_size,
    run_within_object,
):
    """General purpose function to run a task across an image object.

    If run_within_object is True, run the task steps within the image object and return tuple of
    (futures, coords) If run_within_object is False run the task as a function and return a tuple
    of (img, coords)
    """
    raw = image_object.load_step(raw_input_step)
    seg = image_object.load_step(seg_input_step)

    # DELETE
    if seg.shape != raw.shape:
        seg = seg[: raw.shape[0], : raw.shape[1], : raw.shape[2]]

    seg = label(seg)
    regions = find_objects(seg.astype(int))

    results = []
    all_coords = []
    for lab, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        coords, is_edge = pad_slice(coords, padding, seg.shape)
        if not include_edge and is_edge:
            continue
        crop_raw, crop_seg = raw[coords], seg[coords]
        # skip too small seeds
        if np.sum(crop_seg == lab) < min_seed_size:
            continue
        results.append(
            submit(
                run_watershed_task,
                as_task=run_within_object,
                raw=crop_raw,
                seg=crop_seg,
                lab=lab,
                mode=mode,
                is_edge=is_edge,
                erosion=erosion,
                smooth=smooth,
            )
        )
        all_coords.append(coords)

    return results, all_coords, raw.shape


@flow(task_runner=create_task_runner(), log_prints=True)
def run_watershed(
    image_object_paths,
    output_name,
    raw_input_step,
    seg_input_step,
    mode="centroid",
    erosion=None,
    min_seed_size=1000,
    include_edge=True,
    padding=10,
    smooth=False,
):
    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                run_object,
                as_task=not run_within_object,
                image_object=obj,
                raw_input_step=raw_input_step,
                seg_input_step=seg_input_step,
                padding=padding,
                include_edge=include_edge,
                mode=mode,
                erosion=erosion,
                smooth=smooth,
                min_seed_size=min_seed_size,
                run_within_object=run_within_object,
            )
        )

    for object_result, obj in zip(all_results, image_objects):
        if run_within_object:
            # parallelizing within fov
            imgs, coords, shape = object_result
            imgs = [im.result() for im in imgs]
        else:
            # parallelizing across fovs
            imgs, coords, shape = object_result.result()

        seg = merge_instance_segs(imgs, coords, np.zeros(shape).astype(np.uint16))
        output = StepOutput(
            obj.working_dir,
            "run_watershed",
            output_name,
            output_type="image",
            image_id=obj.id,
        )
        output.save(seg)
        obj.add_step_output(output)
        obj.save()
