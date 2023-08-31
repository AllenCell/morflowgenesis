import numpy as np

from prefect import flow, task
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from hydra.utils import get_class

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.step_output import StepOutput
from morflowgenesis.utils.image_object import ImageObject

@task
def project_step(image_object, input_step, step_name, output_name, scale, dtype):
    # image resizing
    img = image_object.load_step(input_step)
    img = np.max(img, 0)
    img = rescale(img, scale, order=0, preserve_range=True)
    img= rescale_intensity(img, out_range=dtype).astype(dtype)
    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name=step_name,
        output_name=f'{output_name}_{input_step}',
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img)


@flow(task_runner=create_task_runner(), log_prints=True)
def max_project(
    image_object_path, step_name, output_name, input_steps, scale=0.25, dtype='numpy.uint8'
):
    image_object = ImageObject.parse_file(image_object_path)
    # skip if already run
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return image_object
    
    if scale > 1.0:
        raise ValueError(f'Scale should be less than 1, got {scale}')

    dtype = get_class(dtype)
    
    input_steps = [input_steps] if isinstance(input_steps, str) else input_steps

    results = []
    for step in input_steps:
        results.append(project_step.submit(image_object, step, step_name, output_name, scale, dtype))
    for r in results:
        image_object.add_step_output(r.result())
    image_object.save()
