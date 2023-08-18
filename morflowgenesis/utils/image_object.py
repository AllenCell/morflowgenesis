import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter


class ImageObject:
    def __init__(self, working_dir, source_path, metadata=None):
        self.run_history = []

        # generally useful metadata
        self.source_path = source_path
        self.C = 0
        self.T = 0
        self.S = None

        # user-defined, workflow-specific metadata
        if metadata is not None:
            for k, v in metadata.items():
                setattr(self, k, v)
        self.id = hashlib.sha224(bytes(source_path + str(metadata), "utf-8")).hexdigest()
        self.working_dir = Path(working_dir)
        self.save_path = Path(self.working_dir / "_ImageObjectStore" / f"{self.id}.pkl")
        self.save_path.parent.mkdir(exist_ok=True, parents=True)

    def step_is_run(self, step_name):
        # for a step to be considered run, must be in history and output must exist
        return step_name in self.run_history and self.get_step(step_name).path.exists()

    def add_step_output(self, output):
        # add output to image object
        step_name = f"{output.step_name}_{output.output_name}"
        setattr(self, step_name, output)

        # don't want to mess up history with reruns
        if "step_name" not in self.run_history:
            self.run_history.append(step_name)

    def get_step(self, step_name):
        # load StepOutput object
        return getattr(self, step_name)

    def load_step(self, step_name):
        # load output from StepOutput object
        step_obj = getattr(self, step_name)
        return step_obj.load_output()

    def save(self):
        # pickle image object
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)


class StepOutput:
    def __init__(self, working_dir, step_name, output_name, output_type, image_id, path=None):
        self.image_id = image_id
        self.step_name = step_name
        self.output_name = output_name
        self.output_type = output_type
        file_extension = ".tif" if output_type == "image" else ".csv"
        self.path = path or working_dir / step_name / output_name / (image_id + file_extension)
        Path(self.path).parent.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        return f"StepOutput({self.step_name}, {self.path}, {self.output_name}, {self.output_type})"

    def load_output(self, path=None):
        path = path or self.path
        if self.output_type == "image":
            return AICSImage(self.path).data.squeeze()
        elif self.output_type == "csv":
            return pd.read_csv(self.path)

    def save(self, data, path=None):
        path = path or self.path
        if self.output_type == "image":
            OmeTiffWriter.save(uri=path, data=data)
        elif self.output_type == "csv":
            data.to_csv(path, index=False)
