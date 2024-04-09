import re
from typing import List

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from morflowgenesis.utils import ImageObject
from aicsshparam.shtools import get_reconstruction_from_coeffs, voxelize_meshes
from aicsimageio.writers import OmeTiffWriter
from morflowgenesis.utils import parallelize_across_images


def pca_bottleneck_shcoeffs(pca, data):
    # shcoeff->pca
    x = pca.transform(data)
    # pca -> shcoeff
    y = pca.inverse_transform(x)
    return y

def shcoeff_to_img(shcoeff, lmax=16):
    shcoeff = shcoeff.reshape(2, lmax+1, lmax+1)
    mesh, _ = get_reconstruction_from_coeffs(shcoeff)
    img, _ = voxelize_meshes([mesh])
    return img

def create_multich_image(data):
    max_shape = np.array([d.shape for d in data]).max(0)
    data = [np.pad(d, [(0, max_shape[0] - d.shape[0]), (0, max_shape[1] - d.shape[1]), (0, max_shape[2] - d.shape[2])]) for d in data]
    data = np.stack(data)
    return data

def overlay_pc_reconstructions(cid, features, pca, feature_columns, apply_names, lmax, save_path):
    print('Overlaying PC reconstructions for', cid)
    pred_df = features.xs(cid, level="CellId")
    shcoeffs = pca_bottleneck_shcoeffs(pca, pred_df.loc[apply_names][feature_columns])
    data = create_multich_image([shcoeff_to_img(shcoeff, lmax) for shcoeff in shcoeffs])
    OmeTiffWriter.save(uri = save_path/f"{cid}.tif", data = data, dimension_order = 'CZYX', channel_names=apply_names)


def visualize_pc_reconstructions(
    image_objects: List[ImageObject],
    output_name: str,
    features_step: str,
    features_regex: str,
    apply_names: List[str],
    lmax: int = 16,
    calculate_name: str = None,
    n_components: int = 10,
    reference_features_path: str = None,
    tags: List[str] = [],
):
    """
    Run PCA on features from calculate_name, and apply to features from apply_names
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to run PCA on
    output_name : str
        Name of output
    feature_step : str
        Step name of calculated features
    features_regex : str
        Regular expression to match feature columns
    calculate_name : str
        Name of features to calculate PCA on
    apply_names : List[str]
        List of names to apply PCA to
    n_components : int
        Number of principal components to use
    tags : List[str]
        [UNUSED] Tags corresponding to concurrency-limits for parallel processing
    """
    if calculate_name in apply_names:
        raise ValueError("Calculate_name cannot be in apply_names")
    
    save_path = image_objects[0].working_dir / 'visualize_pc_reconstruction'/ output_name
    save_path.mkdir(parents=True, exist_ok=True)

    features = pd.concat([obj.load_step(features_step) for obj in image_objects])
    feature_columns = [c for c in features.columns if re.search(features_regex, c)]
    if calculate_name is None:
        assert reference_features_path is not None, "Reference features path must be provided if calculate_name is None"
        print('Using reference features for PCA')
        target_df = pd.read_csv(reference_features_path)
    else:
        target_df = features.xs(calculate_name, level="Name")

    pca = PCA(n_components=n_components)
    pca.fit(target_df[feature_columns])

    apply_names.append(calculate_name)

    parallelize_across_images(
        features.index.get_level_values("CellId").unique(),
        overlay_pc_reconstructions,
        tags,
        data_name="cid",
        features=features,
        pca=pca,
        feature_columns=feature_columns,
        apply_names=apply_names,
        lmax=lmax,
        save_path=save_path,
    )

