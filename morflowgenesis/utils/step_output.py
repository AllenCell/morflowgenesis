import json
from pathlib import Path

import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter


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

    def to_dict(self):
        return {
            "image_id": self.image_id,
            "step_name": self.step_name,
            "output_name": self.output_name,
            "output_type": self.output_type,
            "path": str(self.path)
        }

    def to_json(self, indent=None):
        obj_dict = self.to_dict()
        return json.dumps(obj_dict, indent=indent)

    @classmethod
    def from_dict(cls, obj_dict):
        path = Path(obj_dict["path"])
        instance = cls(
            path.parent.parent,  # working_dir
            obj_dict["step_name"],
            obj_dict["output_name"],
            obj_dict["output_type"],
            obj_dict["image_id"],
            path
        )
        return instance

    @classmethod
    def from_json(cls, json_string):
        obj_dict = json.loads(json_string)
        return cls.from_dict(obj_dict)
