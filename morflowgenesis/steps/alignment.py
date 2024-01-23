import numpy as np
from prefect import flow, task
from skimage.segmentation import find_boundaries

from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner, submit


@task
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


@task
def run_object(
    image_object,
    image_step,
    segmentation_steps,
    run_within_object,
    boundary=False,
):
    results = []
    for step in segmentation_steps:
        results.append(
            submit(
                align,
                as_task=run_within_object,
                image_object=image_object,
                image_step=image_step,
                segmentation_step=step,
                boundary=boundary,
            )
        )
    return image_object, results


@flow(task_runner=create_task_runner(), log_prints=True)
def align_segmentations_to_image(
    image_object_paths, image_step, segmentation_steps, boundary=False
):
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                run_object,
                as_task=not run_within_object,
                image_object=obj,
                image_step=image_step,
                segmentation_steps=segmentation_steps,
                run_within_object=run_within_object,
                boundary=boundary,
            )
        )

    for results in all_results:
        if run_within_object:
            # parallelizing within fov
            object, results = results
            results = [r.result() for r in results]
        else:
            # parallelizing across fovs
            object, results = results.result()
        for result in results:
            object.add_step_output(result)
        object.save()
