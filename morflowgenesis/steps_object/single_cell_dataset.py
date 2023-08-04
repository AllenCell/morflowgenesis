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
# logger = get_run_logger()
from prefect.task_runners import ConcurrentTaskRunner
from morflowgenesis.utils.image_object import StepOutput, Cell


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
    img = (img == label)*255
    return img.astype(np.uint8)

def reshape(img,  z_res, xy_res, qcb_res, order = 0):
    return rescale(
        img,
        (z_res / qcb_res, xy_res / qcb_res, xy_res / qcb_res),
        order=order,
        preserve_range=True,
        # multichannel=False,
    ).astype(np.uint8)

def get_roi_from_regionprops(bbox, img_shape):
    # determine crop roi
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

def mask_images(raw_images, seg_images,label,splitting_column, bbox):
    mask = crop(seg_images[splitting_column]==label, bbox)
    raw_images= {k: crop(v, bbox) for k, v in raw_images.items()}
    seg_images = {k: mask * crop(v>0, bbox) for k, v in seg_images.items()}
    return raw_images, seg_images

def get_volumes(seg_images):
    seg_volumes = {f'{k}_volume': np.sum(v) for k, v in seg_images.items()}
    return seg_volumes

@task
def extract_cell(image_object, raw_images, seg_images , prop, splitting_step, raw_steps, seg_steps,  qcb_res, z_res, xy_res,tp=None, upload_fms=False, dataset_name = 'morphogenesis', tracking_df=None):
    # get bounding box based on splitting_column
    roi, bbox = get_roi_from_regionprops(prop.bbox,  seg_images[splitting_step].shape)

    cell_id = hashlib.sha224(
            bytes(image_object.id + str(roi), "utf-8")
    ).hexdigest()
   
    df = {
        'CellId': cell_id,
        'workflow':'standard_workflow',
        'roi': str(roi),
        "scale_micron": str([qcb_res]*3),
        "centroid_x": (bbox[2] + bbox[5]) // 2,
        "centroid_y": (bbox[1] + bbox[4]) // 2,
        "centroid_z": (bbox[0] + bbox[3]) // 2,
    }
    if tracking_df is not None:
        tracking_df = tracking_df[tracking_df.label_image == prop.labl]
        df.update({
            "frame": tp,
            "track_id": tracking_df.track_id.iloc[0],
            "lineage_id": tracking_df.lineage_id.iloc[0],
            "label_img": tracking_df.label_img.iloc[0],
            "is_outlier": tracking_df.is_outlier.iloc[0],
            "parent": tracking_df.parent.iloc[0],
            "daughter": tracking_df.daughter.iloc[0],
            "edge_cell": tracking_df.edge_cell.iloc[0],
        })
    raw_images, seg_images = mask_images(raw_images, seg_images, prop.label, splitting_step, roi)
    volumes = get_volumes(seg_images)
    df.update(volumes)

    img_paths = {step_name : image_object.get_step(step_name).path for step_name in raw_steps+ seg_steps}
    df.update(img_paths)
    df = pd.DataFrame([df])

    thiscell_path = image_object.working_dir / 'single_cell_dataset' / str(cell_id)
    if os.path.isdir(thiscell_path):
        rmtree(thiscell_path)
    Path(thiscell_path).mkdir(parents=True, exist_ok=True)

    raw_images = {k: reshape(v, z_res, xy_res, qcb_res, order = 3) for k, v in raw_images.items()}
    seg_images = {k: reshape(v, z_res, xy_res, qcb_res, order = 0) for k, v in seg_images.items()}

    name_dict ={}
    for output_type, data in zip(['raw', 'seg'], [raw_images, seg_images]):
        fns = sorted(data.keys())
        imgs = np.asarray([data[k] for k in fns])
        channel_names = [f"crop_{output_type}_{c}" for c in fns]
        save_path = thiscell_path / f"{output_type}.tiff"
        OmeTiffWriter().save(uri = save_path, data =imgs, dimension_order = 'CZYX', channel_names = channel_names)

        FMS_meta = {"id": cell_id, "path": save_path}
        if upload_fms:
            crop_FMS = upload_file(
                fms_env="prod",
                file_path=save_path,
                intended_file_name=cell_id + f"_{output_type}.tiff",
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
def single_cell_dataset(image_object,step_name, output_name, splitting_step, raw_steps, seg_steps, tracking_step = None, xy_res=0.108, z_res=0.29, qcb_res=0.108, upload_fms=False):
    assert splitting_step in seg_steps, 'Splitting step must be included in `seg_steps`'
    seg_images = {
        step_name: image_object.load_step(step_name) for step_name in seg_steps
    }
    raw_images = {
        step_name: image_object.load_step(step_name) for step_name in raw_steps
    }
    raw_images = {
        k: rescale_intensity(v, out_range=np.uint8).astype(np.uint8) for  k, v in raw_images.items() 
    }
    # assert all images same shape
    assert len(set([v.shape for v in raw_images.values()]+[v.shape for v in seg_images.values()])) == 1, "images are not same shape"

    # get regionprops of segmentations
    seg_props = regionprops(seg_images[splitting_step].astype(int))

    tp = image_object.T

    tracking_df = None
    if tracking_step is not None:
        tracking_step_obj = image_object.load_step(tracking_step)
        tracking_df = tracking_step_obj.load_output(tracking_step_obj.output_path/'edges.csv')
        tracking_df = tracking_df[tracking_df.time_index] ==  tp

    results =[]
    for prop in seg_props:
        results.append(extract_cell.submit( image_object, raw_images, seg_images , prop, splitting_step, raw_steps, seg_steps,  qcb_res, z_res, xy_res,tp=tp, upload_fms=False, dataset_name = 'morphogenesis', tracking_df=None))
        break
    df = pd.concat([r.result() for r in results])
    csv_output_path = image_object.working_dir / 'single_cell_dataset'/ f'{image_object.id}.csv'
    step_output = StepOutput(image_object.working_dir, step_name,output_name, output_type='csv', image_id = image_object.id, path = csv_output_path)
    image_object.add_step_output(step_output)
    image_object.save()
    df.to_csv(csv_output_path,  index=False)
    return image_object
