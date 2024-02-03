from pathlib import Path

import numpy as np
import pandas as pd
from prefect import task
from scipy.ndimage import find_objects
from scipy.signal import medfilt
from timelapsetracking import csv_to_nodes
from timelapsetracking.tracks import add_connectivity_labels
from timelapsetracking.tracks.edges import add_edges
from timelapsetracking.viz_utils import visualize_tracks_2d

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


def str_to_array(s):
    elements = s[1:-1].strip().split()
    return np.array(list(map(int, elements)))


def create_regionprops_csv(obj, input_step, output_name):
    inst_seg = obj.get_step(input_step).load_output()
    save_path = Path(f"{obj.working_dir}/tracking/{output_name}/{obj.id}_regionprops.csv")
    if save_path.exists():
        out = pd.read_csv(save_path)
        out["img_shape"] = out["img_shape"].apply(str_to_array)
        return out

    timepoint = obj.metadata["T"]

    field_shape = np.array(inst_seg.shape, dtype=int)
    regions = find_objects(inst_seg.astype(int))

    data = []
    for lab, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        min_coors = np.asarray([s.start for s in coords])
        max_coors = np.asarray([s.stop for s in coords])

        is_edge = np.any(np.logical_or(min_coors == 0, max_coors == field_shape))

        centroid = [(s.start + s.stop) // 2 for s in coords]
        row = pd.DataFrame(
            [
                {
                    "CellLabel": lab,
                    "Timepoint": timepoint,
                    "Centroid_z": centroid[0],
                    "Centroid_y": centroid[1],
                    "Centroid_x": centroid[2],
                    "Volume": np.sum(inst_seg[coords] == lab),
                    "Edge_Cell": is_edge,
                    "img_shape": field_shape,
                }
            ]
        )
        data.append(row)
    out = pd.concat(data)
    out.to_csv(save_path, index=False)
    return out


def find_outliers_by_volume(vol, thresh=0.10, pad_size=15, kernel=9):
    """detect errors in instance segmentation through changes in volume."""
    # TODO this makes outliers easier at the end of the movie
    # normalize data relative to minimum size
    vol = vol / np.min(vol)
    vol_pad = np.pad(vol, pad_size, mode="edge")  # , stat_length=3)

    # median filter to remove outliers
    vol_filt = medfilt(vol_pad, kernel)[pad_size:-pad_size]

    # get absolute change between real and filtered volumes
    change = abs(vol - vol_filt)

    # find locations of deviations above threshold
    outliers = np.argwhere(change > thresh).flatten()
    return outliers


def get_outliers(df):
    df = df.sort_values(by="time_index")
    df.reset_index(inplace=True)

    # find outliers and note their locations in track
    outliers = find_outliers_by_volume(df["volume"].values)
    df.loc[outliers, "is_outlier"] = True

    # note if track has outliers and whichg timpoints come afterwards
    if len(outliers) > 0:
        df.loc[outliers[0] :, "past_outlier"].iloc[outliers[0] :] = True
        df["has_outlier"] = True
    return df


def get_cell_state(df):
    idxs = df.index.values.tolist()
    # skip lineages with 1 timepoint
    if len(idxs) == 1:
        return df

    for idx in idxs:
        # if a pair was detected during instance seg, classify as
        # daughter cell
        df.loc[idx, "daughter"] = df.loc[idx, "has_pair"]

        # if cell has 2 edges out, or the next cell is a daughter,
        # classify as parent cell
        next_idx = idxs.index(idx) + 1
        if next_idx < len(idxs):
            df.loc[idx, "parent"] = (
                df.iloc[next_idx]["has_pair"] or len(eval(df.loc[idx]["out_list"])) > 1
            ) and not df.loc[idx, "daughter"]

        # if not parent or daughter, cell is migrating normally
        df.loc[idx, "normal_migration"] = not (df.loc[idx]["daughter"] or df.loc[idx]["parent"])

    return df


def outlier_detection(df_track):
    # add new columns to tracking table
    cols = ["is_outlier", "has_outlier", "past_outlier", "parent", "daughter"]
    for col_name in cols:
        df_track[col_name] = False
    df_track["normal_migration"] = True

    # Perform outlier detection on tracking results and annotate
    print("Outlier detections")
    grouped = df_track.groupby("track_id")
    df_track = grouped.apply(get_outliers).reset_index(drop=True)

    print("Cell state")
    # Estimate cell state based on tracking/instance seg results
    grouped = df_track.groupby("lineage_id")
    df_track = grouped.apply(get_cell_state).reset_index(drop=True)
    return df_track


@task()
def track(regionprops, working_dir, output_name, edge_thresh_dist=75):
    output_dir = working_dir / "tracking" / output_name
    tracking_output = StepOutput(
        working_dir,
        step_name="tracking",
        output_name=output_name,
        output_type="csv",
        image_id="",
        path=output_dir / "outliers.csv",
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

    outliers = outlier_detection(df_edges)
    outliers.to_csv(f"{output_dir}/outliers.csv")

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


def tracking(image_object_paths, tags, run_type, output_name, input_step):
    image_objects = [ImageObject.parse_file(p) for p in image_object_paths]
    Path(f"{image_objects[0].working_dir}/tracking/{output_name}").mkdir(
        parents=True, exist_ok=True
    )
    if _do_tracking(image_objects, output_name):
        # create centroid/volume csv
        _, regionprops = parallelize_across_images(image_objects, create_regionprops_csv, tags=tags, input_step=input_step, output_name=output_name)

        output = track(pd.concat(regionprops), image_objects[0].working_dir, output_name)
        for obj in image_objects:
            obj.add_step_output(output)
            obj.save()
