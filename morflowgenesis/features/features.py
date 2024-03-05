import numpy as np
import torch
import vtk
from aicsshparam import shparam, shtools
from skimage.measure import label
from torchmetrics.classification import BinaryF1Score


class SurfaceArea:
    def __init__(self, sigma=2):
        self.sigma = sigma
        self.massp = vtk.vtkMassProperties()

    def __call__(self, img):
        try:
            mesh, _, _ = shtools.get_mesh_from_image(image=img, sigma=self.sigma)
        except ValueError as e:
            print(e)
            return {"mesh_vol": np.nan, "mesh_sa": np.nan}

        self.massp.SetInputData(mesh)
        self.massp.Update()
        return {"mesh_vol": self.massp.GetVolume(), "mesh_sa": self.massp.GetSurfaceArea()}


class Centroid:
    def __call__(self, img):
        z, y, x = np.where(img)
        return {"centroid": (z.mean(), y.mean(), x.mean())}


class AxisLengths:
    def __call__(self, img):
        # alignment adds a channel dimension
        img, _ = shtools.align_image_2d(img.copy(), compute_aligned_image=True)
        _, _, y, x = np.where(img)
        return {"width": np.ptp(y) + 1, "length": np.ptp(x) + 1}


class HeightPercentile:
    def __call__(self, img):
        z, _, _ = np.where(img)
        return {"height_percentile": np.percentile(z, 99.9) - np.percentile(z, 0.1)}


class Volume:
    def __call__(self, img):
        return {"volume": np.sum(img)}


class Height:
    def __call__(self, img):
        z, _, _ = np.where(img)
        return {"height": z.max() - z.min()}


class NPieces:
    def __call__(self, img):
        return {"n_pieces": len(np.unique(label(img))) - 1}


class SHCoeff:
    def __init__(self, lmax=16, sigma=2):
        """
        Parameters
        ----------
        lmax: int
            Maximum spherical harmonics degree
        sigma: float
            Sigma for smoothing before mesh generation
        """
        self.lmax = lmax
        self.sigma = sigma

    def __call__(self, img, transform_params=None):
        """
        Calculate spherical harmonics coefficients for a segmentation
        Parameters
        ----------
        img: np.ndarray
            3D image data
        transform_params: Optional[List]
            List of transform parameters for alignment generated by previous alignment step
        return_transform: bool
            Whether to return the transform parameters for use in subsequent calls
        """
        alignment_2d = True
        if transform_params is not None:
            img = shtools.apply_image_alignment_2d(img, transform_params[-1])[0]
            alignment_2d = False
        try:
            (coeffs, _), (_, _, _, transform_params) = shparam.get_shcoeffs(
                image=img, lmax=self.lmax, alignment_2d=alignment_2d, sigma=self.sigma
            )
        except ValueError as e:
            print(e)
            return {"shcoeff": None}
        coeffs.update({"transform_params": transform_params})
        return coeffs


class F1Score:
    def __call__(self, img, reference):
        return {
            "f1": BinaryF1Score()(
                torch.from_numpy(img > 0), torch.from_numpy(reference > 0)
            ).item()
        }
