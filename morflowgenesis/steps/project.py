import numpy as np
from hydra.utils import get_class
from prefect import flow, task
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    create_task_runner,
    submit,
    to_list,
)


@task
def project_task(
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


@task
def run_object(
    image_object,
    input_steps,
    output_name,
    scale,
    dtype,
    run_within_object,
    project_type="max",
    project_slice=None,
    axis=None,
    intensity_rescale_ranges=None,
):
    """General purpose function to run a task across an image object.

    If run_within_object is True, run the task steps within the image object and return a list of
    futures of output objects If run_within_object is False run the task as a function and return a
    list of output objects
    """
    results = []
    for i, step in enumerate(input_steps):
        results.append(
            submit(
                project_task,
                as_task=run_within_object,
                image_object=image_object,
                input_step=step,
                output_name=output_name,
                scale=scale,
                dtype=dtype,
                project_type=project_type,
                project_slice=project_slice,
                axis=axis,
                intensity_rescale_range=intensity_rescale_ranges[i]
                if intensity_rescale_ranges is not None
                else None,
            )
        )
    return results


@flow(task_runner=create_task_runner(), log_prints=True)
def project(
    image_object_paths,
    output_name,
    input_steps,
    scale=1.0,
    dtype="numpy.uint8",
    project_type="max",
    project_slice=None,
    axis=None,
    intensity_rescale_ranges=None,
):
    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    input_steps = to_list(input_steps)
    dtype = get_class(dtype)

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                run_object,
                as_task=not run_within_object,
                image_object=obj,
                input_steps=input_steps,
                output_name=output_name,
                scale=scale,
                dtype=dtype,
                project_type=project_type,
                project_slice=project_slice,
                axis=axis,
                intensity_rescale_ranges=intensity_rescale_ranges,
                run_within_object=run_within_object,
            )
        )

    for object_result, obj in zip(all_results, image_objects):
        if run_within_object:
            # parallelizing within fov
            object_result = [r.result() for r in object_result]
        else:
            # parallelizing across fovs
            object_result = object_result.result()
        for output in object_result:
            obj.add_step_output(output)
