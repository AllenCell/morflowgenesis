import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from cyto_dl.api import CytoDLModel
from distributed import get_worker
from prefect import task

from morflowgenesis.utils import ImageObject, StepOutput


def download_mlflow_model(
    run_id: str,
    save_path: str,
    checkpoint_path="checkpoints/val/loss/best.ckpt",
    tracking_uri: str = "https://mlflow.a100.int.allencell.org",
):
    if (save_path / checkpoint_path).exists():
        print("Checkpoint exists! Skipping download...")
        return save_path / checkpoint_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.artifacts.download_artifacts(
        run_id=run_id, tracking_uri=tracking_uri, artifact_path=checkpoint_path, dst_path=save_path
    )
    return save_path / checkpoint_path


@task
def load_model(
    image_objects,
    output_name,
    input_step,
    config_path,
    overrides,
    run_id=None,
    checkpoint_path="checkpoints/val/loss/best.ckpt",
):
    working_dir = image_objects[0].working_dir

    mlflow_ckpt_path = None
    if run_id is not None:
        save_path = working_dir / run_id
        save_path.mkdir(exist_ok=True, parents=True)
        mlflow_ckpt_path = download_mlflow_model(run_id, save_path, checkpoint_path)
        overrides.update({"ckpt_path": mlflow_ckpt_path})

    save_dir = working_dir / "run_cytodl"
    if output_name is not None:
        save_dir = save_dir / output_name
    save_dir.mkdir(exist_ok=True, parents=True)

    model = CytoDLModel()
    model.load_config_from_file(config_path)

    overrides.update(
        {
            "data.data": [
                {model.cfg.model.x_key: str(im.get_step(input_step).path)} for im in image_objects
            ],
            "model.save_dir": str(save_dir),
            "paths.output_dir": str(save_dir),
            "paths.work_dir": str(save_dir),
        }
    )
    model.override_config(overrides)
    return model


@task(retries=3, retry_delay_seconds=[10, 60, 120])
def run_evaluate(model):
    # suppress error if not running with dask cuda cluster
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(get_worker().name)
    except ValueError:
        assert (
            "CUDA_VISIBLE_DEVICES" in os.environ
        ), "CUDA_VISIBLE_DEVICES must be set if not using `dask_gpu` task runner!"
    return model.predict()


def run_cytodl(
    image_objects: List[ImageObject],
    input_step: str,
    config_path: str,
    output_name: Optional[str] = None,
    overrides: Dict[str, Any] = {},
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = "checkpoints/val/loss/best.ckpt",
    n_partitions: int = 2,
    tags: List[str] = [],
):
    """Wrapper function to run cytoDL on a list of image objects. Note that the output will be
    saved to `working_dir/run_cytodl/head_name` for each output task head of your model.

    Parameters
    ----------
    image_objects:
        image objects to run
    input_step : str
        Name of step to use as input data
    config_path : str
        Path to base config file
    output_name : Optional[str], optional
        Name of output. If none is provided, the model output heads will be used as output name. If provided, the head name will be concatenated to the output name in the format `output_name/head_name`
    overrides : Dict[str, Any], optional
        Dictionario of overrides to apply to config, by default {}
    run_id : Optional[str], optional
        MLFlow run ID to download model from, by default None
    checkpoint_path : Optional[str], optional
        Path to checkpoint to download from MLFlow, by default "checkpoints/val/loss/best.ckpt"
    n_partitions : int, optional
        Number of partitions to split the image objects into for parallel processing. By default 2, this should match the number of GPUs available to your dask_gpu worker.
    tags : List[str], optional
        [UNUSED] Tags corresponding to concurrency-limits for parallel processing
    """
    # split image objects into partitions for parallel running and submit jobs
    n_objects_per_partition = math.ceil(len(image_objects) / n_partitions)
    results = []
    for i in range(n_partitions):
        start = i * n_objects_per_partition
        end = (i + 1) * n_objects_per_partition
        if i == n_partitions - 1:
            end = len(image_objects)
        obj = image_objects[start:end]
        if len(obj) == 0:
            continue

        model = load_model(
            obj,
            output_name,
            input_step,
            Path(config_path),
            overrides,
            run_id,
            checkpoint_path,
        )
        results.append(run_evaluate.submit(model))

    # gather io maps
    out = []
    for r in results:
        out += r.result()[2]

    # save outputs to image objects
    if out is None:
        return
    for batch in out:
        # match model predictions to image objects by input filename
        for input_filename, output_dict in batch.items():
            for i in range(len(image_objects)):
                if input_filename == str(image_objects[i].get_step(input_step).path):
                    # rename output heads to match output_name if provided
                    for head, save_path in output_dict.items():
                        output = StepOutput(
                            image_objects[0].working_dir,
                            step_name="run_cytodl",
                            output_name=head if output_name is None else f"{output_name}/{head}",
                            output_type="image",
                            image_id=image_objects[i].id,
                        )
                        output.path = save_path
                        image_objects[i].add_step_output(output)
                        image_objects[i].save()
