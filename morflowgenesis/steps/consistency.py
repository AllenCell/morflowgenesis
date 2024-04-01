from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from morflowgenesis.utils import ImageObject


def error_plot(feats, feat_name: str, save_dir: Path):
    feats.quantile(np.arange(0, 1, 0.01)).plot()
    plt.xlabel("Percentile")
    plt.ylabel("Absolute Error")
    plt.title(
        f"{feat_name}: [50, 99]% quantile error {round(feats.median(), 2)}, {round(feats.quantile(0.99),2)}, n={len(feats)}"
    )
    plt.savefig(save_dir / f"{feat_name}_error_plot.png")
    plt.close()

def consistency_validation(
    image_objects: List[ImageObject],
    manifest_step: str,
    features_list: List[str],
    tags: List[str] = [],
    min_track_length: int = 10,
):
    """
    Run PCA on features from calculate_name, and apply to features from apply_names
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to run PCA on
    output_name : str
        Name of output
    manifest_step : str
        Manifest step name with features and tracking
    features_list : List[str]
    tags : List[str]
        [UNUSED] Tags corresponding to concurrency-limits for parallel processing
    min_track_length : int
        Minimum track length to consider
    """
    features = image_objects[0].load_step(manifest_step)
    save_dir = image_objects[0].working_dir / "consistency_validation"
    save_dir.mkdir(exist_ok=True)
    # filter out tracks that are too short
    features = features.groupby("track_id").filter(lambda x: len(x) >= min_track_length)
    for feat in features_list:
        abs_error = features.groupby("track_id")[feat].apply(
            lambda x: np.abs(x - x.median()) * (0.108**3)
        )
        error_plot(abs_error, feat, save_dir)
