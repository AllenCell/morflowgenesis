import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import json
import pandas as pd
import hashlib
from pathlib import Path
import pickle

class ImageObject:
    def __init__(self, working_dir, source_path, metadata):
        self.run_history =[]
        self.source_path = source_path
        for k, v in metadata.items():
            setattr(self, k, v)
        self.id = hashlib.sha224(bytes(source_path +str(metadata), "utf-8")).hexdigest()
        self.working_dir = Path(working_dir)
        self.save_path = Path(self.working_dir/'_ImageObjectStore'/f"{self.id}.pkl")
        self.save_path.parent.mkdir(exist_ok=True, parents=True)

    def current_step(self):
        return  getattr(self, self.run_history[-1])
    
    def step_is_run(self, step_name):
        return step_name in self.run_history and self.get_step(step_name).path.exists()

    def add_step_output(self, output):
        step_name = f'{output.step_name}_{output.output_name}'
        setattr(self, step_name, output)

        # don't want to mess up history with reruns
        if 'step_name' not in self.run_history:
            self.run_history.append(step_name)

    def get_step(self, step_name):
        return  getattr(self, step_name)
    
    def load_step(self, step_name):
        step_obj = getattr(self, step_name)
        return step_obj.load_output()

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)

class Cell(ImageObject):
    def __init__(self, parent_image, roi):
        self.parent_image = parent_image
        self.working_dir = parent_image.working_dir
        self.id = hashlib.sha224(
            bytes(parent_image.id + str(roi), "utf-8")
        ).hexdigest()
        self.run_history = parent_image.run_history
        self.save_path = Path(self.working_dir/'_CellObjectStore'/f"{self.id}.pkl")
        self.save_path.parent.mkdir(exist_ok=True, parents=True)

class StepOutput:
    def __init__(self, working_dir, step_name, output_name, output_type, image_id, path = None,):
        # what to do about multiple outputs?
        # be able to overwrite paths afterward
        self.image_id =image_id
        self.step_name = step_name
        self.output_name = output_name
        self.output_type = output_type
        file_extension = '.tif' if output_type == 'image' else '.csv'
        self.path = path or working_dir/step_name/output_name/(image_id+file_extension) 
        Path(self.path).parent.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        return f"StepOutput({self.step_name}, {self.path}, {self.output_name}, {self.output_type})"

    def load_output(self, path = None):
        path = path or self.path
        if self.output_type == 'image':
            return AICSImage(self.path).data.squeeze()
        elif self.output_type == 'csv':
            return pd.read_csv(self.path)
    
    def save(self, data, path = None):
        path = path or self.path
        if self.output_type == 'image':
            OmeTiffWriter.save(uri = path, data=data)   
        elif self.output_type == 'csv':
            data.to_csv(path , index=False)