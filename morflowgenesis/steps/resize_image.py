import os

from prefect import flow
from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner
from skimage.transform import rescale, resize

from morflowgenesis.utils.image_object import StepOutput

DASK_ADDRESS = os.environ.get("DASK_ADDRESS", None)


@flow(task_runner=DaskTaskRunner(address=DASK_ADDRESS) if DASK_ADDRESS else SequentialTaskRunner())
def run_resize(
    image_object, step_name, output_name, input_step, output_shape=None, scale=None, order=0
):
    # skip if already run
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
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
