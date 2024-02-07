import numpy as np
from hydra.utils import get_class
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    parallelize_across_images,
    to_list,
)


def run_project(
    image_object,
    input_step,
    output_name,
    scale,
    dtype,
    project_type="max",
    project_slice=None,
    axis=None,
    intensity_rescale_range=None,
):
    # image resizing
    img = image_object.load_step(input_step)
    if project_type == "max":
        img = np.max(img, 0)
    if project_type == "slice":
        assert isinstance(project_slice, int), "project_slice must be an integer"
        if axis is None or axis == 0:
            img = img[project_slice]
        elif axis == 1:
            img = img[:, project_slice]
        elif axis == 2:
            img = img[:, :, project_slice]

    img = rescale(img, scale, order=0, preserve_range=True, anti_aliasing=False)
    if intensity_rescale_range is not None:
        if not isinstance(intensity_rescale_range, str):
            intensity_rescale_range = tuple(intensity_rescale_range)
        img = rescale_intensity(img, in_range=intensity_rescale_range, out_range=dtype).astype(
            dtype
        )
    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name="project",
        output_name=f"{output_name}_{input_step}_{project_type}_{project_slice}_{axis}",
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img)
    return output


def project(
    image_objects,
    tags,
    output_name,
    input_steps,
    scale=1.0,
    dtype="numpy.uint8",
    project_type="max",
    project_slice=None,
    axis=None,
    intensity_rescale_ranges=None,
    run_type=None,
):
    input_steps = to_list(input_steps)
    dtype = get_class(dtype)

    for i, step in enumerate(input_steps):
        parallelize_across_images(
            image_objects,
            run_project,
            tags=tags,
            input_step=step,
            output_name=output_name,
            scale=scale,
            dtype=dtype,
            project_type=project_type,
            project_slice=project_slice,
            axis=axis,
            intensity_rescale_ranges=intensity_rescale_ranges[i],
        )
