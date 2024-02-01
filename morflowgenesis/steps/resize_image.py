from prefect import flow, task
from skimage.transform import rescale as sk_rescale
from skimage.transform import resize as sk_resize

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    create_task_runner,
    submit,
    to_list,
)


@task
def run_resize(image_object, output_name, input_step, output_shape=None, scale=None, order=0):
    # image resizing
    img = image_object.load_step(input_step)
    input_dtype = img.dtype
    if output_shape is not None:
        img = sk_resize(img, output_shape, order=order, preserve_range=True, anti_aliasing=False)
    elif scale is not None:
        img = sk_rescale(img, scale, order=order, preserve_range=True, anti_aliasing=False)
    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name="resize",
        output_name=f"{output_name}_{input_step}",
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img.astype(input_dtype))
    return output


@task(name="resize")
def run_object(
    image_object,
    output_name,
    input_steps,
    run_within_object,
    output_shape=None,
    scale=None,
    order=0,
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
                run_resize,
                as_task=run_within_object,
                image_object=image_object,
                input_step=step,
                output_name=output_name,
                output_shape=output_shape,
                scale=scale,
                order=order,
            )
        )
    return results


@flow(task_runner=create_task_runner(), log_prints=True)
def resize(image_object_paths, output_name, input_steps, output_shape=None, scale=None, order=0):

    input_steps = to_list(input_steps)

    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                run_object,
                as_task=not run_within_object,
                image_object=obj,
                output_name=output_name,
                input_steps=input_steps,
                output_shape=output_shape,
                scale=scale,
                order=order,
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
