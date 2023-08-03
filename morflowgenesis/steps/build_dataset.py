import logging
import pandas as pd

# from aicsfiles import FileManagementSystem
# from aicsfiles.model import FMSFile

from pathlib import Path
from prefect import task, get_run_logger
# logger = get_run_logger()


class dummy_fms_upload:
    def __init__(self, id, path):
        self.id = id
        self.path = path


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

@task
def fov_upload(row, raw_cols=None, seg_cols=None, tp_col=None, output_path=None, upload_fms=None, dataset_name=None):
    row = row._asdict()
    # get raw image and segmentation
    outpath = Path(output_path)
    outpath.mkdir(parents=True, exist_ok=True)
    csv_name = Path(row[raw_cols[0]]).name.replace(".tiff", ".csv")
    csv_path = outpath / csv_name
    if csv_path.exists():
        # logger.info("csv file already exists")
        return csv_path

    # logger.info("Uploading raw images to FMS")

    input_data = {k: v for k, v in row.items() if k in raw_cols + seg_cols}

    output_data = {tp_col:row[tp_col] }
    for image_type in input_data:
        upload_info = dummy_fms_upload(
                Path(input_data[image_type]).stem,
                input_data[image_type],
            )
        if upload_fms:
            upload_info = upload_file(
                "prod",
                input_data[image_type],
                f"{dataset_name}_{row[tp_col]}_{image_type}.tiff",
                dataset_name,
                "transfer_function",
            )
        output_data[f"{image_type}_id"] = upload_info.id
        output_data[f"{image_type}_path"] = upload_info.path

    pd.DataFrame(output_data, index=[1]).to_csv(csv_path, header=True, index=False)
    # logger.info(f"csv saved to {csv_path}")
    return csv_path
