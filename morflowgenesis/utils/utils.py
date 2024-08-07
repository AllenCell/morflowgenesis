import time

import numpy as np
import tqdm
from omegaconf import ListConfig
from prefect.flows import Flow
from prefect.tasks import Task
from scipy.ndimage import find_objects, label


def submit(task_function, tags=[], name=None, **kwargs):
    name = name or task_function.__name__
    task = Task(task_function, name=name, tags=tags, log_prints=True)
    return task.submit(**kwargs)


def parallelize_across_images(
    data, task_function, tags=[], data_name="image_object", delay=0.5, **kwargs
):
    """data is list of image objects results is list of step outputs, one per image object."""
    results = []
    for d in data:
        kwargs.update({data_name: d})
        results.append(submit(task_function, tags=tags, **kwargs))
        # rate limit task submission to spare postgres db
        if delay > 0:
            time.sleep(delay)
    results = [r.result() for r in results]
    return data, results


def run_flow(flow_function, task_runner, tags, **kwargs):
    flow = Flow(flow_function, task_runner=task_runner)
    kwargs.update({"tags": tags})
    # returns completed, pending, running, and failed state
    return flow._run(**kwargs).type


def to_list(x):
    if isinstance(x, (list, ListConfig)):
        return x
    return [x]


def get_largest_cc(im, mask=None, do_label=False):
    if do_label:
        im, n = label(im)
    else:
        n = im.max()
    if n > 0:
        mask = mask if mask is not None else np.ones_like(im)
        largest_cc = np.argmax(np.bincount((im * mask).flatten())[1:]) + 1
        return (im == largest_cc).astype(np.uint8)
    return im


def pad_coords(s, padding, constraints, include_ch=False, return_edge=True):
    if isinstance(padding, int):
        padding = [padding] * len(s)
    # pad slice by padding subject to image size constraints
    is_edge = False
    new_slice = [slice(None, None)] if include_ch else []
    for slice_part, c, p in zip(s, constraints, padding):
        if slice_part.start == 0 or slice_part.stop >= c:
            is_edge = True
        start = max(0, slice_part.start - p)
        stop = min(c, slice_part.stop + p)
        new_slice.append(slice(start, stop, None))
    if return_edge:
        return tuple(new_slice), is_edge
    return tuple(new_slice)


def extract_objects(img, padding=0, constraints=None, include_ch=False, return_zip=True):
    """takes in labeled image, amount to pad coordinates, and maximum value of coordinates returns
    tuples of (lab, coords, is_edge)"""
    if constraints is None:
        constraints = img.shape
    print("Finding objects")
    regions = find_objects(img)
    labels = []
    rois = []
    edge = []

    for lab, coords in tqdm.tqdm(enumerate(regions, start=1), desc="Extracting objects"):
        if coords is None:
            continue
        labels.append(lab)
        coords, is_edge = pad_coords(
            coords, padding, constraints, include_ch=include_ch, return_edge=True
        )
        rois.append(coords)
        edge.append(is_edge)
    if return_zip:
        return zip(labels, rois, edge)
    return labels, rois, edge
