from aicsshparam import shtools
from prefect import flow, task
from skimage.transform import rescale

from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner


@task
def create_mesh(image_object, step_name, output_name, seg_step, resize):
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
        else mech_output_directory / f"{image_object.id}.vtk"
    )

    step_output = StepOutput(
        image_object.working_dir,
        step_name,
        output_name,
        "image",
        image_id=image_object.id,
        path=save_path,
    )
    shtools.save_polydata(mesh, str(step_output.path))
    return step_output


@flow(task_runner=create_task_runner(), log_prints=True)
def mesh(
    image_object_path,
    step_name,
    output_name,
    segmentation_steps,
    resize=0.2,
):
    image_object = ImageObject.parse_file(image_object_path)

    results = []
    for seg_step in segmentation_steps:
        results.append(create_mesh.submit(image_object, step_name, output_name, seg_step, resize))
    for r in results:
        image_object.add_step_output(r.result())
    image_object.save()
