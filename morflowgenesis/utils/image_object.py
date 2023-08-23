import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from .step_output import StepOutput


class ImageObject(BaseModel):
    working_dir: Path
    source_path: str
    save_path: Path
    id: str
    metadata: Optional[Dict] = None
    C: Optional[int] = 0
    T: Optional[int] = 0
    S: Optional[int] = None

    run_history: List[str] = []
    _steps: Dict[str, StepOutput] = {}

    def __init__(
        self, working_dir, source_path, metadata=None, C=0, T=0, S=None, run_history=[], _steps={}
    ):
        super().__init__(
            working_dir=working_dir,
            source_path=source_path,
            metadata=metadata,
            C=C,
            T=T,
            S=S,
            run_history=run_history,
            _steps=_steps,
        )
        self.run_history = []

        # generally useful metadata
        self.source_path = source_path
        self.C = 0
        self.T = 0
        self.S = None
        # user-defined, workflow-specific metadata
        self.metadata = metadata

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
        Path(self.save_path).write_text(self.json())
