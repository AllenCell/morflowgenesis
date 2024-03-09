import subprocess
from typing import List

from morflowgenesis.utils import ImageObject


def run_command(
    commands: List[str],
    image_objects: List[ImageObject],
    tags: List[str] = [],
):
    """
    Run shell command
    ----------
    commands : List[str]
        List of shell commands to run
    image_objects : List[ImageObject]
        List of ImageObjects to run PCA on
    tags : List[str]
        [UNUSED] Tags corresponding to concurrency-limits for parallel processing
    """
    for command in commands:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Get the output
        stdout, stderr = process.communicate()

        # Decode the output
        stdout = stdout.decode()
        stderr = stderr.decode()

        # Print the output
        print("Output:", stdout)
        if stderr:
            print("Error:", stderr)
