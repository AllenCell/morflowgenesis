import numpy as np
from prefect import flow, task
from scipy.ndimage import binary_dilation, binary_erosion, find_objects
from skimage.measure import label
from skimage.segmentation import watershed

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


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
    # other objects are good seeds
    bg = np.logical_and(seg > 0, seg != lab)
    bg = binary_erosion(bg, iterations=5).astype(int)

    # boundaries are good seeds due to padding. we have to be careful because the object could
    # be touching the edge of the fov
    border_mask = np.ones_like(seg)
    border_mask[1:-1, 1:-1, 1:-1] = 0
    border_mask[binary_dilation(seg == lab, iterations=5)] = 0

    return np.logical_or(bg, border_mask)


@task
def run_watershed_task(raw, seg, lab, mode, is_edge, erosion=5):
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
    seed[bg_seed] = 2

    seg = watershed(raw, seed)
    seg = seg == 1

    border_mask = np.ones_like(seg)
    border_mask[1:-1, 1:-1, 1:-1] = 0
    # if something is edge here, it is intended to be included, here we filter out non-edge segmentations that touch the border
    # (indicating the watershed has escaped the intended objects), or really large edge cells
    if (not is_edge and np.sum(border_mask * seg) > 1000) or (is_edge and np.mean(seg) > 0.5):
        return np.zeros_like(seg)

    return seg


def merge_instance_segs(segs, coords, img):
    lab = np.uint16(1)
    for c, s in zip(coords, segs):
        img[c] += s.astype(np.uint16) * lab
        lab += 1
    return img


@flow(task_runner=create_task_runner(), log_prints=True)
def run_watershed(
    image_object_path,
    step_name,
    output_name,
    raw_input_step,
    seg_input_step,
    mode="centroid",
    erosion=None,
    min_seed_size=1000,
    include_edge=True,
    padding=10,
):
    image_object = ImageObject.parse_file(image_object_path)
    output = StepOutput(
        image_object.working_dir,
        step_name,
        output_name,
        output_type="image",
        image_id=image_object.id,
    )
    if output.path.exists():
        return

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
        results.append(run_watershed_task.submit(crop_raw, crop_seg, lab, mode, is_edge, erosion))
        all_coords.append(coords)
    results = [r.result() for r in results]

    seg = merge_instance_segs(results, all_coords, np.zeros_like(seg).astype(np.uint16))

    output.save(seg)
    image_object.add_step_output(output)
    image_object.save()
