import hashlib
import json
import os
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
from aicsimageio.writers import OmeTiffWriter
from prefect import flow, task
from scipy.ndimage import find_objects
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.step_output import StepOutput
from morflowgenesis.utils.image_object import ImageObject


def upload_file(
    fms_env: str,
    file_path: Path,
    intended_file_name: str,
    prov_input_file_id: str,
    prov_algorithm: str,
):
    """Upload a file located on the Isilon to FMS."""
    raise NotImplementedError

def reshape(img, z_res, xy_res, qcb_res, order=0):
    return rescale(
        img,
        (z_res / qcb_res, xy_res / qcb_res, xy_res / qcb_res),
        order=order,
        preserve_range=True,
        # multichannel=False,
    ).astype(np.uint8)

def centroid_from_slice(slicee):
    return [(s.start + s.stop) // 2 for s in slicee]

def roi_from_slice(slicee):
    return ",".join([f"{s.start},{s.stop}" for s in slicee])


@task
def extract_cell(
    image_object,
    output_name,
    raw_images,
    seg_images,
    roi,
    coords,
    lab,
    raw_steps,
    seg_steps,
    qcb_res,
    z_res,
    xy_res,
    upload_fms=False,
    dataset_name="morphogenesis",
    tracking_df=None,
):
    # prepare metadata for csv
    roi = roi_from_slice(roi)
    cell_id = hashlib.sha224(bytes(image_object.id + roi, "utf-8")).hexdigest()

    centroid = centroid_from_slice(coords)

    df = {
        "CellId": cell_id,
        "workflow": "standard_workflow",
        "roi": roi,
        "scale_micron": str([qcb_res] * 3),
        "centroid_x": centroid[2],
        "centroid_y": centroid[1],
        "centroid_z": centroid[0],
    }
    if tracking_df is not None:
        tracking_df = tracking_df[tracking_df.label_image == lab]
        df.update(
            {
                "frame": tracking_df.time_index.iloc[0],
                "track_id": tracking_df.track_id.iloc[0],
                "lineage_id": tracking_df.lineage_id.iloc[0],
                "label_img": tracking_df.label_img.iloc[0],
                "is_outlier": tracking_df.is_outlier.iloc[0],
                "parent": tracking_df.parent.iloc[0],
                "daughter": tracking_df.daughter.iloc[0],
                "edge_cell": tracking_df.edge_cell.iloc[0],
            }
        )

    img_paths = {
        step_name: image_object.get_step(step_name).path for step_name in raw_steps + seg_steps
    }
    df.update(img_paths)
    df = pd.DataFrame([df])

    # remove cell folder if it exists
    thiscell_path = image_object.working_dir / "single_cell_dataset" / output_name / str(cell_id)
    if os.path.isdir(thiscell_path):
        rmtree(thiscell_path)
    Path(thiscell_path).mkdir(parents=True, exist_ok=True)

    # anisotropic resize
    raw_images = {k: reshape(v, z_res, xy_res, qcb_res, order=3) for k, v in raw_images.items()}
    seg_images = {k: reshape(v, z_res, xy_res, qcb_res, order=0) for k, v in seg_images.items()}

    # save out raw and segmentation single cell images
    name_dict = {}
    for output_type, data in zip(["raw", "seg"], [raw_images, seg_images]):
        fns = sorted(data.keys())
        # possible that there is no raw or segmented image available for this cell
        if len(fns) == 0:
            continue
        # stack segmentation/raw images into multichannel image
        imgs = np.asarray([data[k] for k in fns])
        channel_names = [f"crop_{output_type}_{c}" for c in fns]
        save_path = thiscell_path / f"{output_type}.tiff"
        OmeTiffWriter().save(
            uri=save_path, data=imgs, dimension_order="CZYX", channel_names=channel_names
        )

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
        name_dict[f"crop_{output_type}"] = channel_names
    df["name_dict"] = json.dumps(name_dict)
    return df

@task
def pad_slice(s, padding, constraints):
    # pad slice by padding subject to image size constraints
    new_slice = []
    for slice_part, c in zip(s, constraints):
        start = max(0, slice_part.start - padding)
        stop = min(c, slice_part.stop + padding)
        new_slice.append(slice(start, stop, None))
    return tuple(new_slice)

@task
def mask_images(raw_images, seg_images, label, splitting_column, coords, mask=True):
    # use masking segmentation to crop out non-cell regions
    if mask:
        mask_img = seg_images[splitting_column][coords] == label
    
    raw_images = {k: v[coords] for k, v in raw_images.items()}
    seg_images = {k: mask_img * (v > 0)[coords] if mask else ( v > 0)[coords] for k, v in seg_images.items()}
    return raw_images, seg_images

@task
def load_images(image_object, splitting_step, seg_steps, raw_steps, seg_steps_rename, raw_steps_rename):
    '''
    load images into dictionaries with (optionally) renamed names as keys
    '''
    assert splitting_step in seg_steps, "Splitting step must be included in `seg_steps`"

    if seg_steps_rename is None:
        seg_images = {step_name: image_object.load_step(step_name) for step_name in seg_steps}
    else:
        assert len(seg_steps) == len(seg_steps_rename), 'seg_steps and seg_steps_rename must have same length'
        seg_images = {step_rename: image_object.load_step(step_name) for step_name, step_rename in zip(seg_steps, seg_steps_rename)}
        # update splitting step as well
        splitting_step = seg_steps_rename[seg_steps.index(splitting_step)]

    if raw_steps_rename is None:
        raw_images = {step_name: image_object.load_step(step_name) for step_name in raw_steps}
    else:
        assert len(raw_steps) == len(raw_steps_rename), 'seg_steps and seg_steps_rename must have same length'
        raw_images = {step_rename: image_object.load_step(step_name) for step_name, step_rename in zip(raw_steps, raw_steps_rename)}

    raw_images = {
        k: rescale_intensity(v, out_range=np.uint8).astype(np.uint8) if v.dtype != np.uint8 else v
        for k, v in raw_images.items()
    }
    # check all images same shape
    assert (
        len(set([v.shape for v in raw_images.values()] + [v.shape for v in seg_images.values()]))
        == 1
    ), "images are not same shape"

    return raw_images, seg_images, splitting_step

@flow(task_runner=create_task_runner(), log_prints=True)
def single_cell_dataset(
    image_object_path,
    step_name,
    output_name,
    splitting_step,
    raw_steps,
    seg_steps,
    raw_steps_rename = None,
    seg_steps_rename = None,
    tracking_step=None,
    xy_res=0.108,
    z_res=0.29,
    qcb_res=0.108,
    padding= 10,
    mask=True,
    upload_fms=False,
):
    image_object = ImageObject.parse_file(image_object_path)

    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return image_object

    raw_images, seg_images, splitting_step = load_images(image_object, splitting_step, seg_steps, raw_steps, seg_steps_rename, raw_steps_rename)

    # find objects in segmentation
    regions = find_objects(seg_images[splitting_step].astype(int))

    # load timepoint's tracking data if available
    tracking_df = None
    if tracking_step is not None:
        tracking_df = image_object.load_step(tracking_step)
        tracking_df = tracking_df[tracking_df.time_index == image_object.metadata['T']]

    padded_coords = []
    unpadded_coords = []
    labs = []
    for lab, coords in enumerate(regions, start=1):
        if coords is None:
            continue
        padded_coords.append(pad_slice.submit(coords, padding, seg_images[splitting_step].shape))
        unpadded_coords.append(coords)
        labs.apend(lab)
    padded_coords = [pc.result() for pc in padded_coords]

    crop_images = []
    for lab, coords in zip(labs, padded_coords):
        crop_images.append(mask_images.submit(raw_images, seg_images, lab, splitting_step, coords, mask=mask))
    crop_images = [im.result() for im in crop_images]


    results = []
    for padded_coord, unpadded_coord, crop_img, lab in zip(padded_coords, unpadded_coords, crop_images, labs):
        crop_raw_images, crop_seg_images = crop_img
        results.append(
            extract_cell.submit(
                image_object,
                output_name,
                crop_raw_images,
                crop_seg_images,
                padded_coord,
                unpadded_coord,
                lab,
                raw_steps,
                seg_steps,
                qcb_res,
                z_res,
                xy_res,
                upload_fms=False,
                dataset_name="morphogenesis",
                tracking_df=tracking_df,
            )
        )

    # results = []
    # for lab, coords in enumerate(regions, start=1):
    #     if coords is None:
    #         continue
    #     padded_coords = pad_slice(coords, padding, seg_images[splitting_step].shape)
    #     # do cropping serially to avoid memory blow up
    #     crop_raw_images, crop_seg_images = mask_images(
    #         raw_images, seg_images, lab, splitting_step, padded_coords, mask=mask
    #     )
    #     results.append(
    #         extract_cell.submit(
    #             image_object,
    #             output_name,
    #             crop_raw_images,
    #             crop_seg_images,
    #             padded_coords,
    #             coords,
    #             lab,
    #             raw_steps,
    #             seg_steps,
    #             qcb_res,
    #             z_res,
    #             xy_res,
    #             upload_fms=False,
    #             dataset_name="morphogenesis",
    #             tracking_df=None,
    #         )
    #     )

    df = pd.concat([r.result() for r in results])
    csv_output_path = (
        image_object.working_dir / "single_cell_dataset" / output_name / f"{image_object.id}.csv"
    )

    step_output = StepOutput(
        image_object.working_dir,
        step_name,
        output_name,
        output_type="csv",
        image_id=image_object.id,
        path=csv_output_path,
    )
    step_output.save(df)

    image_object.add_step_output(step_output)
    image_object.save()
