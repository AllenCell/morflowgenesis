import fire
import yaml
from pathlib import Path

def main(config_path, step_number):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f.read())
    step = data['steps'][step_number]
    image_object_path = str(list((Path(data['working_dir'])/'_ImageObjectStore').glob('*'))[0])

    args_str = ' '.join({f'{k}="{str(v)}",' if isinstance(v, str) else f'{k}={str(v)},' for k, v in step['args'].items()})
    print(f'''\n\nif __name__ == '__main__':\n\t{step['function'].split('.')[-1]}({args_str}step_name='test',image_object_path='{image_object_path}')\n\n''')


if __name__ == '__main__':
    fire.Fire(main)