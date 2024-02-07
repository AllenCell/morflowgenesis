from morflowgenesis.utils import ImageObject
import fire
from pathlib import Path
import os
import shutil
import json
import tqdm

def main(base_dir, steps={}, inplace=False):
    basedir = Path(base_dir)
    image_objects = [ImageObject.parse_file(fn) for fn in tqdm.tqdm((basedir/'_ImageObjectStore/').glob('*'), desc='Loading image objects')]

    name_map= {
        image_object.id: os.path.basename(image_object.source_path) for image_object in image_objects
    }

    # name_map= {
    #     image_object.id: f"T{image_object.metadata['T']:04d}.tif" for image_object in image_objects
    # }


    for step_name, output_name in steps.items():
        for fn in (basedir/step_name/output_name).glob('*'):
            (basedir/step_name/f'{output_name}_renamed').mkdir(exist_ok=True, parents=True)
            if inplace:
                try:
                    os.rename(fn, basedir/step_name/f'{output_name}_renamed'/name_map[fn.stem])
                except:
                    continue
            else:
                try:
                    shutil.copy(fn, basedir/step_name/f'{output_name}_renamed'/name_map[fn.stem])
                except:
                    continue


if __name__ == '__main__':
    fire.Fire(main)


