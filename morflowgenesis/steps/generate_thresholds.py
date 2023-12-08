import numpy as np
from prefect import flow, task
from skimage.measure import label as run_label

from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner, submit


@task
def run_threshold(image_object, img, output_name, thresh, label):
    step = StepOutput(
        working_dir=image_object.working_dir,
        step_name="threshold",
        output_name=f"{output_name}_{thresh}",
        output_type="image",
        image_id=image_object.id,
    )
    out = img > thresh
    if label:
        out = run_label(out)
    step.save(out.astype(np.uint8))
    return step


@task
def run_object(image_object, output_name, input_step, threshold_range, run_within_object, label):
    """General purpose function to run a task across an image object.

    If run_within_object is True, parallelize across thresholds and return a list of futures of
    step objects If run_within_object is False, run thresholds in sequence and return a list of
    step objects
    """

    img = image_object.load_step(input_step)

    results = []
    for thresh in threshold_range:
        results.append(
            submit(
                run_threshold,
                as_task=run_within_object,
                image_object=image_object,
                output_name=output_name,
                img=img,
                thresh=thresh,
                label=label,
            )
        )
    return results


@flow(task_runner=create_task_runner(), log_prints=True)
def threshold(
    image_object_paths,
    output_name,
    input_step,
    start,
    stop,
    step=None,
    n=None,
    label=False,
):
    if step is not None:
        # include end in range
        threshold_range = np.arange(start, stop + 0.1 * step, step)
    elif n is not None:
        threshold_range = np.linspace(start, stop, n)
    else:
        raise ValueError("Either `step` or `n` must be provided")

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
                input_step=input_step,
                threshold_range=threshold_range,
                run_within_object=run_within_object,
                label=label,
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
