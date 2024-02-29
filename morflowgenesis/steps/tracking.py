from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tqdm
from prefect import task
from timelapsetracking import csv_to_nodes
from timelapsetracking.tracks import add_connectivity_labels
from timelapsetracking.tracks.edges import add_edges

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    extract_objects,
    parallelize_across_images,
)


def str_to_array(s):
    elements = s[1:-1].strip().split()
    return np.array(list(map(int, elements)))


def create_regionprops_csv(image_object, input_step, output_name):
    inst_seg = image_object.get_step(input_step).load_output()
    save_path = Path(
        f"{image_object.working_dir}/tracking/{output_name}/{image_object.id}_regionprops.csv"
    )
    if save_path.exists():
        out = pd.read_csv(save_path)
        out["img_shape"] = out["img_shape"].apply(str_to_array)
        out["Edge_Cell"] = out["Edge_Cell"].astype(bool)
        return out

    timepoint = image_object.metadata["T"]

    field_shape = np.array(inst_seg.shape, dtype=int)

    objects = extract_objects(inst_seg)
    data = []
    for lab, coords, _ in tqdm.tqdm(objects):
        cell = inst_seg[coords] == lab
        z, y, x = np.where(cell)
        data.append(
            {
                "CellLabel": lab,
                "Timepoint": timepoint,
                "Centroid_z": np.mean(z) + coords[0].start,
                "Centroid_y": np.mean(y) + coords[1].start,
                "Centroid_x": np.mean(x) + coords[2].start,
                "Volume": np.sum(cell),
                "Edge_Cell": np.any(
                    np.logical_or(
                        np.asarray([s.start for s in coords]) == 0,
                        np.asarray([s.stop for s in coords]) == field_shape,
                    )
                ),
                "img_shape": field_shape,
            }
        )
    print("saving")

    out = pd.DataFrame(data)
    out.to_csv(save_path, index=False)
    return out


@task()
def track(regionprops, working_dir, output_name, edge_thresh_dist=75):
    output_dir = working_dir / "tracking" / output_name
    tracking_output = StepOutput(
        working_dir,
        step_name="tracking",
        output_name=output_name,
        output_type="csv",
        image_id="",
        path=output_dir / "edges.csv",
    )

    meta_dict = {
        "time_index": "Timepoint",
        "index_sequence": "Timepoint",
        "volume": "Volume",
        "label_img": "CellLabel",
        "zyx_cols": ["Centroid_z", "Centroid_y", "Centroid_x"],
        "edge_cell": "Edge_Cell",
    }
    img_shape = regionprops["img_shape"].iloc[0]

    df = csv_to_nodes.csv_to_nodes(regionprops, meta_dict, img_shape)
    df_edges = add_edges(df, thresh_dist=edge_thresh_dist)
    df_edges = add_connectivity_labels(df_edges)

    df_edges.to_csv(f"{output_dir}/edges.csv")
    return tracking_output


def _do_tracking(image_objects, output_name):
    # check if any step does not have tracking output
    run = False
    for obj in image_objects:
        if not obj.step_is_run(f"tracking/{output_name}"):
            run = True
            break
    if not run:
        print(f"Skipping step `tracking/{output_name}`")
    return run


def tracking(image_objects: List[ImageObject], tags: List[str], output_name: str, input_step: str):
    """
    Tracking based on minimizing centroid distance and change in volume between timepoints
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to process
    tags : List[str]
        List of tags to use for parallel processing
    output_name : str
        Name of output
    input_step : str
        Name of step to load images from
    """
    Path(f"{image_objects[0].working_dir}/tracking/{output_name}").mkdir(
        parents=True, exist_ok=True
    )
    if _do_tracking(image_objects, output_name):
        # create centroid/volume csv
        _, regionprops = parallelize_across_images(
            image_objects,
            create_regionprops_csv,
            tags=tags,
            input_step=input_step,
            output_name=output_name,
        )

        output = track(pd.concat(regionprops), image_objects[0].working_dir, output_name)
        for obj in image_objects:
            obj.add_step_output(output)
            obj.save()
