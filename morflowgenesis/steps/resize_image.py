from prefect import get_run_logger, task
from skimage.transform import rescale, resize

from morflowgenesis.utils.image_object import StepOutput

logger = get_run_logger()


@task
def run_resize_task(
    image_object, step_name, output_name, input_step, output_shape=None, scale=None, order=0
):
    # skip if already run
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        logger.info(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return image_object
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
    return image_object


@flow
def run_resize(
    image_object, step_name, output_name, input_step, output_shape=None, scale=None, order=0
):
    run_resize_task(
        image_object,
        step_name,
        output_name,
        input_step,
        output_shape=output_shape,
        scale=scale,
        order=order,
    )
