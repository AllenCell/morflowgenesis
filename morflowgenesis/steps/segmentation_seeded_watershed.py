import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, find_objects, gaussian_filter
from skimage.measure import label
from skimage.segmentation import watershed

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


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
    regions = find_objects(seg.astype(int))

    results = []
    for lab, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        coords, is_edge = pad_slice(coords, padding, seg.shape)
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
    seg = merge_instance_segs(results, regions, np.zeros(raw.shape).astype(np.uint16))
    output = StepOutput(
        image_object.working_dir,
        "run_watershed",
        output_name,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(seg)


def run_watershed(
    image_objects,
    tags,
    output_name,
    raw_input_step,
    seg_input_step,
    mode="centroid",
    erosion=None,
    min_seed_size=1000,
    include_edge=True,
    padding=10,
):
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
