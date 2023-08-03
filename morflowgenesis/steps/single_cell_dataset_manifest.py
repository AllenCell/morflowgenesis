import os
from shutil import rmtree

import pandas as pd
import numpy as np
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
# from aicsfiles import FileManagementSystem
# from aicsfiles.model import FMSFile

from pathlib import Path

import hashlib
from prefect import task,flow, get_run_logger
import json
logger = get_run_logger()
from prefect.task_runners import ConcurrentTaskRunner


pd.set_option("mode.chained_assignment", None)
###############################################################################


def upload_file(
    fms_env: str,
    file_path: Path,
    intended_file_name: str,
    prov_input_file_id: str,
    prov_algorithm: str,
):
    """
    Upload a file located on the Isilon to FMS.
    """

    metadata = {
        "provenance": {
            "input_files": [prov_input_file_id],
            "algorithm": prov_algorithm,
        },
        "custom_metadata": {
            "annotations": [
                {"annotation_id": 153, "values": ["NucMorph"]},  # Program
            ],
        },
    }

    fms = FileManagementSystem(env=fms_env)
    return fms.upload_file(
        file_path, file_type="image", file_name=intended_file_name, metadata=metadata
    )


def crop(img, roi):
    return img[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]


def clean_seg_crop(img, label):
    img = img == label
    img[img > 0] = 255
    return img.astype(np.uint8)


def reshape(img,  z_res, xy_res, qcb_res, order = 0):
    return rescale(
        img,
        (z_res / qcb_res, xy_res / qcb_res, xy_res / qcb_res),
        order=order,
        preserve_range=True,
        # multichannel=False,
    ).astype(np.uint8)


def get_roi_from_regionprops(bboxes, labels, label_img, img_shape):
    # finding roi can be done just in caax (roi>>lamin)
    this_cell_index = labels.index(label_img)

    # determine crop roi
    bbox = bboxes[this_cell_index]
    z_range = (bbox[0], bbox[3])
    y_range = (bbox[1], bbox[4])
    x_range = (bbox[2], bbox[5])

    # pad edges of roi for nicer crop
    roi = [
        max(z_range[0] - 10, 0),
        min(z_range[1] + 10, img_shape[0]),
        max(y_range[0] - 50, 0),
        min(y_range[1] + 50, img_shape[1]),
        max(x_range[0] - 50, 0),
        min(x_range[1] + 50, img_shape[2]),
    ]
    return roi, bbox

def mask_images(img_data,label,splitting_column, bbox):
    mask = crop(img_data[splitting_column]==label, bbox)
    return {k: mask * crop(v>0, bbox) if 'seg' in k else crop(v, bbox) for k, v in img_data.items() }

def get_volumes(masked_images):
    seg_volumes = {f'{k}_volume': np.sum(v) for k, v in masked_images.items() if 'seg' in k}
    return seg_volumes


@task
def extract_cell(row, img_data, img_paths, splitting_column, tp, output_path, qcb_res, z_res, xy_res, upload_fms, dataset_name, file_exists, current_csv, bboxes, labels):
    # get bounding box based on splitting_column
    roi, bbox = get_roi_from_regionprops(bboxes, labels, row.label_img, img_data[splitting_column].shape)
    cell_id = hashlib.sha224(
        bytes(dataset_name + str(tp) + str(roi), "utf-8")
    ).hexdigest()
    # logger.info(f"cell id: {cell_id}")
    if file_exists and str(cell_id) in current_csv["CellId"].values:
        # logger.info("cell already logged")
        return
    df = {
        'CellId': cell_id,
        'workflow':'standard_workflow',
        'roi': str(roi),
        "scale_micron": str([qcb_res]*3),
        # "node_id": row.node_id,
        "T_index": tp,#frame
        "centroid_x": (bbox[2] + bbox[5]) // 2,
        "centroid_y": (bbox[1] + bbox[4]) // 2,
        "centroid_z": (bbox[0] + bbox[3]) // 2,
        # "index_sequence": tp,
        # "in_list": row.in_list,
        # "out_list": row.out_list,
        "track_id": row.track_id,
        "lineage_id": row.lineage_id,
        "label_img": row.label_img,
        "is_outlier": row.is_outlier,
        # "has_outlier": row.has_outlier,
        # "past_outlier": row.past_outlier,
        # "normal_migration": row.normal_migration,
        "parent": row.parent,
        "daughter": row.daughter,
        "edge_cell": row.edge_cell,
    }
    masked_images = mask_images(img_data, row.label_img, splitting_column, roi)
    volumes = get_volumes(masked_images)
    df.update(volumes)
    df.update(img_paths)
    df = pd.DataFrame([df])

    thiscell_path = output_path/str(cell_id)
    if os.path.isdir(thiscell_path):
        rmtree(thiscell_path)
    Path(thiscell_path).mkdir(parents=True, exist_ok=True)

    output_data = {k: reshape(v, z_res, xy_res, qcb_res, order = 'raw' in k) for k, v in masked_images.items()}
    name_dict ={}
    for output_type in ['raw', 'seg']:
        fns = sorted([k for k in output_data.keys() if output_type in k])
        imgs = np.asarray([output_data[k] for k in fns])
        channel_names = [f"crop_{output_type}_{c}" for c in fns]
        save_path = thiscell_path / f"{output_type}.tiff"
        OmeTiffWriter().save(uri = save_path, data =imgs, dimension_order = 'CZYX', channel_names = channel_names)

        FMS_meta = {"id": cell_id, "path": save_path}
        if upload_fms:
            crop_FMS = upload_file(
                fms_env="prod",
                file_path=save_path,
                intended_file_name=cell_id + "_data.tiff",
                prov_input_file_id=dataset_name,
                prov_algorithm="Crop and resize",
            )

            FMS_meta["id"] = crop_FMS.id
            FMS_meta["path"] = crop_FMS.path
        df[f"crop_{output_type}_id"] = FMS_meta["id"]
        df[f"crop_{output_type}_path"] = FMS_meta["path"]
        df[f"channel_names_{output_type}"] = str(channel_names)
        name_dict[f'crop_{output_type}'] = channel_names
    df['name_dict'] = json.dumps(name_dict)
    return df

@flow(task_runner=ConcurrentTaskRunner(), name = 'Single Cell Extraction')
def cell_upload2(fov_path, tracking_csv_path, splitting_column, tp_col, dataset_name, output_path,  xy_res=0.108, z_res=0.29, qcb_res=0.108, upload_fms=False):
    output_path = Path(output_path)
    df_paths = pd.read_csv(fov_path)
    tp = df_paths[tp_col].iloc[0]
    df_tracking = pd.read_csv(tracking_csv_path)
    df_tracking = df_tracking[df_tracking.time_index== tp]
    
    img_paths = {c: df_paths[c].iloc[0] for c in df_paths.columns if 'path' in c}

    img_data = {
        k: AICSImage(v).data.squeeze() for  k, v in img_paths.items()
    }
    img_data = {
        k: rescale_intensity(v, out_range=np.uint8).astype(np.uint8) if 'raw' in k else v for k, v in img_data.items() 
    }

    # assert all images same shape
    assert len(set([v.shape for v in img_data.values()])) == 1, "images are not same shape"

    # get regionprops of segmentations
    seg_props = regionprops(img_data[splitting_column])
    labels = [prop.label for prop in seg_props]
    bboxes = [prop.bbox for prop in seg_props]

    # make output folder and create output csv path if not given
    output_file = f'img_T{tp:04d}.csv'
    csv_output_path = output_path / output_file

    current_csv = None
    file_exists = csv_output_path.exists()
    if file_exists:
        current_csv = pd.read_csv(csv_output_path)
    results =[]
    for row in df_tracking.itertuples():
        results.append(extract_cell.submit(row, img_data, img_paths, splitting_column, tp, output_path, qcb_res, z_res, xy_res, upload_fms, dataset_name, file_exists, current_csv, bboxes, labels))
    logger.info(results[0])
    df = pd.concat([r.result() for r in results])
    df.to_csv(csv_output_path,  index=False)
    return csv_output_path