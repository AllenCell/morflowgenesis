import os
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, task
from scipy.ndimage import find_objects
from timelapsetracking import csv_to_nodes
from timelapsetracking.tracks import add_connectivity_labels
from timelapsetracking.tracks.edges import add_edges
from timelapsetracking.viz_utils import visualize_tracks_2d

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.step_output import StepOutput


@task
def create_regionprops_csv(obj, input_step):
    inst_seg = obj.get_step(input_step).load_output()
    timepoint = obj.T
    # find centroids and volumes for each instance
    data_table = []
    origin = np.zeros((3,), dtype=int)
    field_shape = np.array(inst_seg.shape, dtype=int)
    regions = find_objects(inst_seg)

    for lab, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        min_coors = np.asarray([s.start for s in coords])
        max_coors = np.asarray([s.stop for s in coords])

        is_edge = np.any(
            np.logical_or(np.equal(min_coors, origin), np.equal(max_coors, field_shape))
        )

        centroid = [(s.start + s.stop) // 2 for s in coords]
        row = {
            "CellLabel": lab,
            "Timepoint": timepoint,
            "Centroid_z": centroid[0],
            "Centroid_y": centroid[1],
            "Centroid_x": centroid[2],
            "Volume": np.sum(inst_seg[coords] == lab),
            "Edge_Cell": is_edge,
            "img_shape": field_shape,
        }
        data_table.append(row)

    return pd.DataFrame(data_table)


@task
def track(regionprops, working_dir, step_name, output_name, edge_thresh_dist=75):
    output_dir = working_dir / step_name / output_name
    tracking_output = StepOutput(
        working_dir,
        step_name=step_name,
        output_name=output_name,
        output_type="csv",
        image_id=None,
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

    visualize_tracks_2d(
        df=df_edges,
        shape=(img_shape[-2], img_shape[-1]),
        path_save_dir=Path(f"{output_dir}/visualization_2d/"),
    )

    return tracking_output


def _do_tracking(image_objects, step_name, output_name):
    # check if any step does not have tracking output
    run = False
    for obj in image_objects:
        if not obj.step_is_run(f"{step_name}_{output_name}"):
            run = True
            break
    if not run:
        print(f"Skipping step {step_name}_{output_name}")
        pass
    return run


@flow(task_runner=create_task_runner(), log_prints=True)
def run_tracking(image_objects, step_name, output_name, input_step):
    if not _do_tracking(image_objects, step_name, output_name):
        return image_objects

    # create centroid/volume csv
    tasks = []
    for obj in image_objects:
        tasks.append(create_regionprops_csv.submit(obj, input_step))
    regionprops = pd.concat([task.result() for task in tasks])

    output = track(regionprops, image_objects[0].working_dir, step_name, output_name)
    for obj in image_objects:
        obj.add_step_output(output)
        obj.save()
    return image_objects
