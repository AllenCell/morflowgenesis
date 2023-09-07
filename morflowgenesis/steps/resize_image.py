import os

from prefect import flow
from skimage.transform import rescale, resize

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.step_output import StepOutput
from morflowgenesis.utils.image_object import ImageObject

@flow(task_runner=create_task_runner(), log_prints=True)
def run_resize(
    image_object_path, step_name, output_name, input_step, output_shape=None, scale=None, order=0
):
    image_object = ImageObject.parse_file(image_object_path)
    
    # image resizing
    img = image_object.load_step(input_step)
    input_dtype = img.dtype
    if output_shape is not None:
        img = resize(img, output_shape, order=order, preserve_range=True)
    elif scale is not None:
        img = rescale(img, scale, order=order, preserve_range=True)
    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name=step_name,
        output_name=output_name,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img.astype(input_dtype))
    image_object.add_step_output(output)
    image_object.save()
