import numpy as np
from hydra.utils import get_class
from prefect import flow, task
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


@task
def project_step(image_object, input_step, step_name, output_name, scale, dtype, project_type='max',project_slice=None, axis=None, intensity_rescale_range=None):
    # image resizing
    img = image_object.load_step(input_step)
    if project_type == 'max':
        img = np.max(img, 0)
    if project_type == 'slice':
        assert isinstance(project_slice, int), 'project_slice must be an integer'
        if axis is None or axis == 0:
            img = img[project_slice]
        elif axis == 1:
            img = img[:,project_slice]
        elif axis == 2:
            img = img[:,:,project_slice]
        
    img = rescale(img, scale, order=0, preserve_range=True, anti_aliasing=False)
    if intensity_rescale_range is not None:
        if not isinstance(intensity_rescale_range, str):
            intensity_rescale_range = tuple(intensity_rescale_range)
        img = rescale_intensity(img, in_range = intensity_rescale_range, out_range=dtype).astype(dtype)
    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name=step_name,
        output_name=f"{output_name}_{input_step}_{project_type}_{project_slice}_{axis}",
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img)
    return output


@flow(task_runner=create_task_runner(), log_prints=True)
def project(
    image_object_path, step_name, output_name, input_steps, scale=0.25, dtype="numpy.uint8", project_type='max',project_slice=None, axis=None,
    intensity_rescale_ranges=None
):
    image_object = ImageObject.parse_file(image_object_path)
    dtype = get_class(dtype)

    input_steps = [input_steps] if isinstance(input_steps, str) else input_steps

    results = []
    for i, step in enumerate(input_steps):
        results.append(
            project_step.submit(image_object, step, step_name, output_name, scale, dtype, project_type, project_slice, axis, intensity_rescale_ranges[i] if intensity_rescale_ranges is not None else None)
        )
    for r in results:
        image_object.add_step_output(r.result())
    image_object.save()
