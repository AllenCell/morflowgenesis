from hydra._internal.utils import _locate
from prefect import flow, task

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    create_task_runner,
    submit,
    to_list,
)


@task
def apply_function(image_object, input_step, output_name, ch, function, function_args):
    data = image_object.load_step(input_step)
    if ch is not None:
        data = data[ch]
    function = _locate(function)

    applied = function(data, **function_args)
    output = StepOutput(
        working_dir=image_object.working_dir,
        step_name="array_to_array",
        output_name=f"{output_name}_{input_step}",
        output_type="image",
        image_id=image_object.id,
    )
    output.save(applied)
    return output


@task
def run_object(
    image_object,
    input_steps,
    output_name,
    run_within_object,
    function,
    function_args,
):
    """General purpose function to run a task across an image object.

    If run_within_object is True, run the task steps within the image object and return a list of
    futures of output objects If run_within_object is False run the task as a function and return a
    list of output objects
    """

    results = []
    for step in input_steps:
        results.append(
            submit(
                apply_function,
                as_task=run_within_object,
                image_object=image_object,
                input_step=step,
                output_name=output_name,
                function=function,
                function_args=function_args,
            )
        )
    return results


@flow(task_runner=create_task_runner(), log_prints=True)
def array_to_array(
    image_object_paths, output_name, input_steps, function, ch=None, function_args={}
):
    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    input_steps = to_list(input_steps)

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                run_object,
                as_task=not run_within_object,
                image_object=obj,
                input_steps=input_steps,
                output_name=output_name,
                run_within_object=run_within_object,
                function=function,
                ch=ch,
                function_args=function_args,
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
