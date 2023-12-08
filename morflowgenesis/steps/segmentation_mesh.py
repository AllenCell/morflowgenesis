from aicsshparam import shtools
from prefect import flow, task
from skimage.transform import rescale

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    create_task_runner,
    submit,
    to_list,
)


@task
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


@task
def run_object(
    image_object,
    input_steps,
    output_name,
    resize,
    run_within_object,
):
    """General purpose function to run a task across an image object.

    If run_within_object is True, run the task steps within the image object and return a list of
    futures of output objects If run_within_object is False run the task as a function and return a
    list of output objects
    """
    results = []
    for step in input_steps:
        results.append(
            submit(
                create_mesh,
                as_task=run_within_object,
                image_object=image_object,
                seg_step=step,
                output_name=output_name,
                resize=resize,
            )
        )
    return results


@flow(task_runner=create_task_runner(), log_prints=True)
def mesh(
    image_object_paths,
    output_name,
    input_steps,
    resize=0.2,
):
    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    input_steps = to_list(input_steps)

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                run_object,
                as_task=not run_within_object,
                image_object=obj,
                input_steps=input_steps,
                output_name=output_name,
                resize=resize,
                run_within_object=run_within_object,
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
