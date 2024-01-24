import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
from cyto_dl.api import CytoDLModel
from prefect import flow, task

from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


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
    return model.predict()


@flow(log_prints=True)
def run_cytodl(
    image_object_paths: List[Union[str, Path]],
    input_step: str,
    config_path: str,
    output_name: Optional[str] = None,
    overrides: Dict[str, Any] = {},
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = "checkpoints/val/loss/best.ckpt",
):
    """Wrapper function to run cytoDL on a list of image objects. Note that the output will be
    saved to `working_dir/run_cytodl/head_name` for each output task head of your model.

    Parameters
    ----------
    image_object_paths : List[str]
        List of paths to image objects to run
    input_step : str
        Name of step to use as input data
    config_path : str
        Path to base config file
    overrides : Dict[str, Any], optional
        Dictionario of overrides to apply to config, by default {}
    run_id : Optional[str], optional
        MLFlow run ID to download model from, by default None
    checkpoint_path : Optional[str], optional
        Path to checkpoint to download from MLFlow, by default "checkpoints/val/loss/best.ckpt"
    """
    image_objects = [ImageObject.parse_file(p) for p in image_object_paths]

    model = load_model(
        image_objects,
        output_name,
        input_step,
        Path(config_path),
        overrides,
        run_id,
        checkpoint_path,
    )
    _, _, out = run_evaluate(model)
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
                        image_objects[i].add_step_output(output)
                        image_objects[i].save()
                        shutil.move(str(save_path), str(output.path))

