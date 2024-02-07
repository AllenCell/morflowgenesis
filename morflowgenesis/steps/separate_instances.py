import numpy as np
from prefect import flow, task
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.segmentation import expand_labels

from morflowgenesis.utils import ImageObject, StepOutput, submit


@task
def create_mask(image_object, input_step, output_name, sigma, expand_distance):
    splitting_mask = image_object.load_step(input_step)
    z_slices = splitting_mask.shape[0]
    splitting_mask = np.max(splitting_mask, 0)

    splitting_mask = gaussian(splitting_mask, sigma=sigma)
    splitting_mask = label(splitting_mask > threshold_otsu(splitting_mask))
    splitting_mask = expand_labels(splitting_mask, distance=expand_distance)

    splitting_mask = np.stack([splitting_mask] * z_slices).astype(np.uint16)

    step_output = StepOutput(
        image_object.working_dir, "generate_mask", output_name, "image", image_id=image_object.id
    )
    step_output.save(splitting_mask)
    return step_output


@flow(log_prints=True)
def generate_mask(image_objects, output_name, input_step, sigma=10, expand_distance=50):
    # if only one image is passed, run across objects within that image. Otherwise, run across images

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                create_mask,
                as_task=not run_within_object,
                image_object=obj,
                input_step=input_step,
                output_name=output_name,
                sigma=sigma,
                expand_distance=expand_distance,
            )
        )

    for output, obj in zip(all_results, image_objects):
        if not run_within_object:
            output = output.result()
        obj.add_step_output(output)
        obj.save()
