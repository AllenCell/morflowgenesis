from typing import List, Optional

import numpy as np
from skimage.measure import label as run_label

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


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
    image_object.add_step_output(step)
    image_object.save()


def threshold(
    image_objects: List[ImageObject],
    tags: List[str],
    output_name: str,
    input_step: str,
    start: float,
    stop: float,
    step: Optional[float] = None,
    n: Optional[int] = None,
    label: Optional[bool] = False,
):
    """
    Run range of thresholds on images. Useful for optimizing segmentation thresholds.
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to run threshold on
    tags : List[str]
        Tags corresponding to concurrency-limits for parallel processing
    output_name : str
        Name of output. The threshold used will be appended to this name in the format `output_name/threshold`
    input_step : str
        Step name of input image
    start : float
        Start of threshold range
    stop : float
        End of threshold range
    step : float, optional
        Step size for threshold range. Only used if `n` is not provided
    n : int, optional
        Number of thresholds to use
    label : bool, optional
        Whether to run label on thresholded image
    """
    if step is not None:
        # include end in range
        threshold_range = np.arange(start, stop + 0.1 * step, step)
    elif n is not None:
        threshold_range = np.linspace(start, stop, n)
    else:
        raise ValueError("Either `step` or `n` must be provided")

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
