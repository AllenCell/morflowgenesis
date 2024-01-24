import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from .step_output import StepOutput


class ImageObject(BaseModel):
    working_dir: Path
    source_path: str
    metadata: Optional[Dict] = None
    save_path: Optional[Path] = Path("")
    id: Optional[str] = None
    steps: Optional[Dict[str, StepOutput]] = {}

    def __init__(
        self,
        working_dir: Path,
        source_path: str,
        metadata: Optional[Dict] = None,
        save_path: Optional[Path] = Path(""),
        id: Optional[str] = None,
        steps: Optional[Dict[str, StepOutput]] = {},
    ):
        super().__init__(
            working_dir=Path(working_dir),
            source_path=source_path,
            metadata=metadata,
            steps=steps,
        )
        self.id = id or str(
            hashlib.sha224(bytes(source_path + str(metadata), "utf-8")).hexdigest()
        )
        self.save_path = Path(self.working_dir / "_ImageObjectStore" / f"{self.id}.json")
        self.save_path.parent.mkdir(exist_ok=True, parents=True)

    def step_is_run(self, step_name):
        # for a step to be considered run, must be in history and output must exist
        return step_name in self.steps and self.get_step(step_name).path.exists()

    def add_step_output(self, output):
        # add output to image object
        step_name = f"{output.step_name}_{output.output_name}"
        self.steps[step_name] = output

    def get_step(self, step_name):
        # load StepOutput object
        return self.steps[step_name]

    def load_step(self, step_name):
        # load output from StepOutput object
        step_obj = self.steps[step_name]
        return step_obj.load_output()

    def save(self):
        Path(self.save_path).write_text(self.json())
