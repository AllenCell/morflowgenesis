from prefect import task, get_run_logger, flow
from prefect.task_runners import ConcurrentTaskRunner
from skimage.measure import regionprops
import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

from timelapsetracking import csv_to_nodes
from timelapsetracking.tracks.edges import add_edges
from timelapsetracking.tracks import add_connectivity_labels
from timelapsetracking.viz_utils import visualize_tracks_2d
from morflowgenesis.utils.image_object import StepOutput


@task
def create_regionprops_csv(obj, input_step):
    data_table = []
    # find centroids and volumes for each instance
    centroids = dict()
    vol = dict()
    edges = dict()

    origin = np.zeros((3,), dtype=int)

    inst_seg = obj.get_step(input_step).load_output()
    timepoint = obj.T

    field_shape = np.array(inst_seg.shape, dtype=int)

    label_info = regionprops(inst_seg)
    for instance_label in label_info:
        centroids[instance_label.label] = instance_label.centroid
        vol[instance_label.label] = instance_label.area
        obj_idxs = instance_label.coords
        min_coors = np.min(obj_idxs, axis=0) 
        max_coors = np.max(obj_idxs, axis=0 )

        edges[instance_label.label] = np.any(
            np.logical_or(
                np.equal(min_coors, origin), np.equal(max_coors, field_shape - 1)
            )
        )
  
    for seg_label in tqdm.tqdm(vol.keys()):
        if seg_label == 0:
            continue
        row = {
            "CellLabel": seg_label,
            "Timepoint": timepoint,
            'Centroid_z':centroids[seg_label][0],
            'Centroid_y':centroids[seg_label][1],
            'Centroid_x':centroids[seg_label][2],
            "Volume": vol[seg_label],
            "Edge_Cell": edges[seg_label],
            'img_shape':  field_shape,
        }
        data_table.append(row)
    return pd.DataFrame(data_table)

@task
def track(regionprops,  working_dir, step_name, output_name, edge_thresh_dist=75):
    output_dir= working_dir/step_name/output_name
    tracking_output = StepOutput(working_dir, step_name=step_name, output_name=output_name, output_type='csv', image_id = None, path = output_dir/'edges.csv')

    meta_dict= {
        'time_index': 'Timepoint',
        'index_sequence': 'Timepoint',
        'volume': 'Volume',
        'label_img': 'CellLabel',
        'zyx_cols': ['Centroid_z', 'Centroid_y', 'Centroid_x'],
        'edge_cell': 'Edge_Cell',
    }
    img_shape = regionprops['img_shape'].iloc[0]

    df = csv_to_nodes.csv_to_nodes(regionprops,  meta_dict, img_shape)
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
    logger = get_run_logger()
    run = False
    for obj in image_objects:
        if not obj.step_is_run(f'{step_name}_{output_name}'):
            run = True
            break
    if not run:
        logger.info(f'Skipping step {step_name}_{output_name}')
        pass
    return run

@flow(task_runner=ConcurrentTaskRunner())
def run_tracking(image_objects, step_name, output_name, input_step):
    if not _do_tracking(image_objects, step_name, output_name):
        return image_objects
    
    tasks = []
    for obj in image_objects:
        tasks.append(create_regionprops_csv.submit(obj, input_step))
    regionprops = pd.concat([task.result() for task in tasks])

    output = track(regionprops, image_objects[0].working_dir, step_name, output_name)
    for obj in image_objects:
        obj.add_step_output(output)
        obj.save()
    return image_objects

