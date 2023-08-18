import hashlib
import json
import pickle
from pathlib import Path

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

from .step_output import StepOutput


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
        self._steps = {}

    def step_is_run(self, step_name):
        # for a step to be considered run, must be in history and output must exist
        return step_name in self.run_history and self.get_step(step_name).path.exists()

    def add_step_output(self, output):
        # add output to image object
        step_name = f"{output.step_name}_{output.output_name}"
        self._steps[step_name] = output

        # don't want to mess up history with reruns
        if "step_name" not in self.run_history:
            self.run_history.append(step_name)

    def get_step(self, step_name):
        # load StepOutput object
        return self._steps[step_name]

    def load_step(self, step_name):
        # load output from StepOutput object
        step_obj = getattr(self, step_name)
        return step_obj.load_output()

    def save(self):
        # pickle image object
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)

    def to_dict(self):
        # Convert ImageObject instance to a dictionary
        obj_dict = {
            "run_history": self.run_history,
            "source_path": str(self.source_path),
            "C": self.C,
            "T": self.T,
            "S": self.S,
            "id": self.id,
            "working_dir": str(self.working_dir),
            "save_path": str(self.save_path),
            "_steps": {k: v.to_json() for k, v in self._steps},
        }
        return obj_dict

    def to_json(self, indent=None):
        # Convert ImageObject instance to a JSON string
        obj_dict = self.to_dict()
        return json.dumps(obj_dict, indent=indent)

    @classmethod
    def from_dict(cls, obj_dict):
        # Create an ImageObject instance from a dictionary
        instance = cls(obj_dict["working_dir"], obj_dict["source_path"])
        instance.run_history = obj_dict["run_history"]
        instance.C = obj_dict["C"]
        instance.T = obj_dict["T"]
        instance.S = obj_dict["S"]
        instance.id = obj_dict["id"]
        instance.save_path = Path(obj_dict["save_path"])
        instance._steps = {k: StepOutput.from_json(v) for k, v in obj_dict["_steps"].items()}
        return instance

    @classmethod
    def from_json(cls, json_string):
        # Create an ImageObject instance from a JSON string
        obj_dict = json.loads(json_string)
        return cls.from_dict(obj_dict)
