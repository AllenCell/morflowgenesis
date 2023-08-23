import numpy as np
from prefect import task
from scipy.ndimage import binary_erosion, find_objects
from skimage.measure import label
from skimage.segmentation import watershed

from morflowgenesis.utils.image_object import StepOutput


def pad_slice(s, padding, constraints):
    # pad slice by padding subject to image size constraints
    new_slice = []
    for slice_part, c in zip(s, constraints):
        start = max(0, slice_part.start - padding)
        stop = min(c, slice_part.stop + padding)
        new_slice.append(slice(start, stop, None))
    return tuple(new_slice)


def get_largest_cc(im):
    im = label(im)
    largest_cc = np.argmax(np.bincount(im.flatten())[1:]) + 1
    return im == largest_cc


@task
def run_watershed_task(raw, seg, lab, mode, erosion=5):
    if mode == "centroid":
        seed = np.zeros_like(seg)
        z, y, x = np.where(seg == lab)
        seed[np.mean(z), np.mean(y), np.mean(x)] = 1
    elif mode == "erosion" and erosion is not None:
        seed = binary_erosion(seg == lab, iterations=erosion)
        if np.max(seed) == 0:
            return np.zeros_like(seg)
        seed = get_largest_cc(seed).astype(int)
        # use eroded segmentation from other objects in image as background seeds
        bg = np.logical_and(seg > 0, seg != lab)
        bg = binary_erosion(bg, iterations=erosion).astype(int)
        seed[bg == 1] = 2
        # background seed in case of no other objects, guaranteed to be background
        # from padding
        seed[0, 0, 0] = 3
    else:
        raise ValueError(f"Unknown mode {mode}, valid options are centroid or erosion")

    seg = watershed(raw, seed)

    seg[seg > 1] = 0

    if np.mean(seg) > 0.9:
        # likely bad watershed
        return np.zeros_like(seg)
    return seg


def merge_instance_segs(segs, coords, img):
    lab = np.uint16(1)
    for c, s in zip(coords, segs):
        img[c] += s.astype(np.uint16) * lab
        lab += 1
    return img


def run_watershed(
    image_object,
    step_name,
    output_name,
    raw_input_step,
    seg_input_step,
    mode="centroid",
    erosion=None,
):
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return image_object

    raw = image_object.load_step(raw_input_step)
    seg = image_object.load_step(seg_input_step)
    regions = find_objects(seg.astype(int))

    results = []
    all_coords = []
    for lab, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        coords = pad_slice(coords, 10, raw.shape)
        crop_raw, crop_seg = raw[coords], seg[coords]
        results.append(run_watershed_task(crop_raw, crop_seg, lab, mode))
        all_coords.append(coords)
    results = [r.result() for r in results]

    seg = merge_instance_segs(results, all_coords, np.zeros_like(seg))
    output = StepOutput(
        image_object.output_dir,
        step_name,
        output_name,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(seg)
    image_object.add_step_output(output)
    image_object.save()
    return image_object
