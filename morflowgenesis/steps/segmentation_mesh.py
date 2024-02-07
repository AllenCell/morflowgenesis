from aicsshparam import shtools
from skimage.transform import rescale

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    parallelize_across_images,
    to_list,
)


def create_mesh(image_object, output_name, seg_step, resize):
    seg = image_object.load_step(seg_step)

    # Reduce the image size and make volume isotropic
    seg = rescale(seg > 0, resize)
    mesh, _, _ = shtools.get_mesh_from_image(seg, sigma=0, lcc=False, translate_to_origin=False)

    mesh_output_directory = image_object.working_dir / "mesh" / output_name / seg_step

    # rename to make movie creation easy
    timepoint = image_object.metadata.get("T")
    save_path = (
        mesh_output_directory / f"{image_object.source_path}_T{timepoint:04d}.vtk"
        if timepoint is not None
        else mesh_output_directory / f"{image_object.id}.vtk"
    )

    step_output = StepOutput(
        image_object.working_dir,
        "mesh",
        f"{output_name}_{seg_step}",
        "image",
        image_id=image_object.id,
        path=save_path,
    )
    shtools.save_polydata(mesh, str(step_output.path))
    return step_output


def mesh(
    image_objects,
    tags,
    output_name,
    input_steps,
    resize=0.2,
    run_type=None,
):
    input_steps = to_list(input_steps)
    for step in input_steps:
        parallelize_across_images(
            image_objects, create_mesh, output_name=output_name, seg_step=step, resize=resize
        )
