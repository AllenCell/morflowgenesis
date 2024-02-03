import numpy as np
from prefect import flow, task
from skimage.measure import label as run_label

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    parallelize_across_images,
    submit,
)


@task
def run_threshold(image_object, input_step, output_name, thresh, label):
    img = image_object.load_step(input_step)
    step = StepOutput(
        working_dir=image_object.working_dir,
        step_name="threshold",
        output_name=f"{output_name}/{thresh}",
        output_type="image",
        image_id=image_object.id,
    )
    out = img > thresh
    if label:
        out = run_label(out)
    step.save(out.astype(np.uint8))
    return step


@flow(log_prints=True)
def threshold(
    image_object_paths,
    tags,
    output_name,
    input_step,
    start,
    stop,
    step=None,
    n=None,
    label=False,
    run_type=None,
):
    if step is not None:
        # include end in range
        threshold_range = np.arange(start, stop + 0.1 * step, step)
    elif n is not None:
        threshold_range = np.linspace(start, stop, n)
    else:
        raise ValueError("Either `step` or `n` must be provided")

    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]

    # TODO this loads each image per threshold instead of loading once and applying all thresholds
    for thresh in threshold_range:
        parallelize_across_images(
            image_objects,
            run_threshold,
            tags,
            output_name=output_name,
            input_step=input_step,
            thresh=thresh,
            label=label,
        )
