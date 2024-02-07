import numpy as np
from skimage.segmentation import find_boundaries

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


def align(
    image_object,
    image_step,
    segmentation_step,
    boundary=False,
):
    img = image_object.load_step(image_step)
    seg = image_object.load_step(segmentation_step)

    assert len(img.shape) == len(seg.shape) == 3, "Image and segmentation must be 3D"
    if boundary:
        seg = find_boundaries(seg)
    # crop to z with segmentation
    z, _, _ = np.where(seg > 0)
    mask = seg[z.min() : z.max()]

    # ensure same xy size
    minimum_shape = np.minimum(img.shape[-2:], mask.shape[-2:])
    img = img[:, : minimum_shape[0], : minimum_shape[1]]
    mask = mask[:, : minimum_shape[0], : minimum_shape[1]]

    # sliding window to find z where segmentation covers most signal
    out = []
    for i in range(0, img.shape[0] - mask.shape[0]):
        out.append(np.sum(img[i : i + mask.shape[0]] * (mask > 0)))
    best_start_z = np.argmax(out)
    new_seg = np.stack([np.zeros(mask.shape[-2:])] * img.shape[0])
    new_seg[best_start_z : best_start_z + mask.shape[0]] = seg[z.min() : z.max()]

    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name="align",
        output_name=segmentation_step,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(new_seg)
    return output


def align_segmentations_to_image(
    image_objects, image_step, segmentation_steps, boundary=False
):
    for step in segmentation_steps:
        parallelize_across_images(
            image_objects, align, image_step=image_step, segmentation_step=step, boundary=boundary
        )
