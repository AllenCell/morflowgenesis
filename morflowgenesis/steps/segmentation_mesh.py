from typing import List, Optional

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
        f"{output_name}/{seg_step}",
        "image",
        image_id=image_object.id,
        path=save_path,
    )
    shtools.save_polydata(mesh, str(step_output.path))
    image_object.add_step_output(step_output)
    image_object.save()


def mesh(
    image_objects: List[ImageObject],
    tags: List[str],
    output_name: str,
    seg_steps: List[str],
    resize: Optional[float] = 0.2,
):
    """Create a mesh from a segmentation image using shtools
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to create mesh from
    tags : List[str]
        Tags corresponding to concurrency-limits for parallel processing
    output_name : str
        Name of output. The seg step name will be appended to this name in the format `output_name/seg_step`
    seg_steps : List[str]
        Step names of input segmentation images
    resize : float, optional
        Scale factor to resize the image before creating the mesh
    """
    seg_steps = to_list(seg_steps)
    for step in seg_steps:
        parallelize_across_images(
            image_objects, create_mesh, output_name=output_name, seg_step=step, resize=resize
        )
