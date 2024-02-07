from omegaconf import ListConfig
from prefect.flows import Flow
from prefect.tasks import Task
from scipy.ndimage import label
import numpy as np

def submit(task_function, tags=[], name = None,**kwargs):
    name = name or task_function.__name__
    task = Task(task_function, name=name, tags=tags, log_prints=True)
    return task.submit(**kwargs)


def parallelize_across_images(
    data, task_function, tags=[], data_name="image_object", create_output=True, **kwargs
):
    """data is list of image objects results is list of step outputs, one per image object."""
    results = []
    for d in data:
        kwargs.update({data_name: d})
        results.append(submit(task_function, tags=tags, **kwargs))
    results = [r.result() for r in results]
    if create_output:
        for d, r in zip(data, results):
            d.add_step_output(r)
            d.save()
    return data, results


def run_flow(flow_function, task_runner,  tags, **kwargs):
    flow = Flow(flow_function, task_runner=task_runner)
    kwargs.update({ "tags": tags})
    # returns completed, pending, running, and failed state
    return flow._run(**kwargs).type


def to_list(x):
    if isinstance(x, (list, ListConfig)):
        return x
    return [x]


def get_largest_cc(im, label = True):
    im, n= label(im)
    if n > 0:
        largest_cc = np.argmax(np.bincount(im.flatten())[1:]) + 1
        return im == largest_cc
    return im


def pad_coords(s, padding, constraints, include_ch=False, return_edge=True):
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

#extract_objects