import numpy as np
from prefect import flow

from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.segmentation import expand_labels

from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


@flow(log_prints=True)
def generate_mask(
    image_object_path, step_name, output_name, input_step, sigma=10
):
    image_object = ImageObject.parse_file(image_object_path)
    splitting_mask = image_object.load_step(input_step)
    z_slices = splitting_mask.shape[0]
    splitting_mask = np.max(splitting_mask, 0)

    splitting_mask = gaussian(splitting_mask, sigma=sigma)
    splitting_mask = label(splitting_mask > threshold_otsu(splitting_mask))
    splitting_mask = expand_labels(splitting_mask, distance= 50)

    splitting_mask = np.stack([splitting_mask]*z_slices).astype(np.uint16)

    step_output = StepOutput(
        image_object.working_dir, step_name, output_name, "image", image_id = image_object.id
    )
    step_output.save(splitting_mask)
    image_object.add_step_output(step_output)
    image_object.save()
