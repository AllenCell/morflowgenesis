from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from pydantic import BaseModel


class StepOutput(BaseModel):
    working_dir: Path
    step_name: str
    output_name: str
    output_type: Literal["image", "csv"]
    image_id: str
    path: Path = None
    index_col: Optional[Union[List, str]] = None

    def __init__(
        self, working_dir, step_name, output_name, output_type, image_id, path=None, index_col=None
    ):
        super().__init__(
            working_dir=working_dir,
            step_name=step_name,
            output_name=output_name,
            output_type=output_type,
            image_id=image_id,
            path=path,
            index_col=index_col,
        )

        file_extension = ".tif" if output_type == "image" else ".csv"
        self.path = Path(
            path or working_dir / step_name / output_name / (image_id + file_extension)
        )
        self.path.parent.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        return f"StepOutput({self.step_name}, {self.path}, {self.output_name}, {self.output_type})"

    def load_output(self, path=None):
        path = path or self.path
        if self.output_type == "image":
            return BioImage(self.path).data.squeeze()
        elif self.output_type == "csv":
            return pd.read_csv(self.path, index_col=self.index_col)

    def save(self, data, path=None):
        path = path or self.path
        if self.output_type == "image":
            OmeTiffWriter.save(uri=path, data=data)
        elif self.output_type == "csv":
            data.to_csv(path)
