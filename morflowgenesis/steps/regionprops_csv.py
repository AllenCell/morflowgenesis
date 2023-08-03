from prefect import task, get_run_logger
from skimage.measure import regionprops
import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from aicsimageio import imread




@task
def create_regionprops_csv(row, path_col, tp_col,save_dir):
    logger = get_run_logger()

    row =row._asdict()
    data_table = []
    # find centroids and volumes for each instance
    centroids = dict()
    vol = dict()
    edges = dict()

    origin = np.zeros((2,), dtype=int)
    inst_seg = imread(row[path_col]).squeeze()
    timepoint = row[tp_col]
    field_shape = np.array([inst_seg.shape[1], inst_seg.shape[2]], dtype=int)

    label_info = regionprops(inst_seg)
    for instance_label in tqdm.tqdm(label_info):
        centroids[instance_label.label] = instance_label.centroid
        vol[instance_label.label] = instance_label.area

        obj_idxs = instance_label.coords
        min_coors = np.min(obj_idxs, axis=0)[1:]
        max_coors = np.max(obj_idxs, axis=0)[1:]

        edges[instance_label.label] = np.any(
            np.logical_or(
                np.equal(min_coors, origin), np.equal(max_coors, field_shape - 1)
            )
        )

    logger.info("building cell csv")

    for seg_label in tqdm.tqdm(vol.keys()):
        if seg_label == 0:
            continue
        row = {
            "CellLabel": seg_label,
            "Pair": False,
            "Timepoint": timepoint,
            "Centroid": str(centroids[seg_label])[1:-1],
            "Volume": vol[seg_label],
            "Edge_Cell": edges[seg_label],
        }
        data_table.append(row)

    save_dir= Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir/f'img_T{timepoint:04d}_region_props.csv'
    pd.DataFrame(data_table).to_csv(save_path, index=False)

    return save_path