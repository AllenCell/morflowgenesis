from prefect import flow, task
import numpy as np

from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput
from skimage.measure import label as run_label



@task
def run_threshold(image_object, img, step_name, output_name, thresh, label):
    step = StepOutput(
        working_dir = image_object.working_dir,
        step_name = step_name,
        output_name = f'{output_name}_{thresh}',
        output_type = 'image',
        image_id = image_object.id
    )
    out = img>thresh
    if label:
        out = run_label(out)
    step.save(out.astype(np.uint8))
    return step

@flow(log_prints=True)
def threshold(
    image_object_path, step_name, output_name, input_step, start, stop, step=None, n=None, label=False
):
    image_object = ImageObject.parse_file(image_object_path)
    img = image_object.load_step(input_step)

    if step is not None:
        #include end in range
        threshold_range = np.arange(start, stop + 0.1*step, step)
    elif n is not None:
        threshold_range = np.linspace(start, stop, n)
    else:
        raise ValueError('Either `step` or `n` must be provided')

    steps = []
    for thresh in threshold_range:
        steps.append(run_threshold.submit(image_object, img, step_name, output_name, thresh, label))
    for step in steps:
        image_object.add_step_output(step.result())
    image_object.save()

