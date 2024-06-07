#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:  # noqa: FURB101
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bumpversion>=0.6.0",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r>=0.3.0",
    "pytest-runner>=5.2",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.4.3",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "bioio>=1.0.1",
    "bioio-czi",
    "bioio-ome-tiff",
    "bioio-tifffile",
    "bioio-ome-zarr",
    "aicsshparam>=0.1.10",
    "camera-alignment-core==1.0.5",
    "cyto_dl==0.1.5",
    "dask>=2023.3.1",
    "hydra-core>=1.3.2",
    "matplotlib>=3.7.2",
    "mlflow>=2.5.0",
    "numpy>=1.24.4",
    "omegaconf>=2.3.0",
    "pandas>=1.5.3",
    "prefect==2.14.20",
    "prefect_dask>=0.2.5",
    "pydantic==1.10.12",
    "python-slugify>=8.0.1",
    "pyshtools>=4.10.3",
    "PyYAML>=6.0.1",
    "scikit-learn",
    "scipy>=1.9.1",
    "scikit-image>=0.19.2",
    "timelapsetracking @ git+ssh://git@github.com/aics-int/aics-timelapse-tracking.git@b905e0f5bbbc84c9b61c2496009ae49295a0b4ab",
    "vtk>=9.2.6",
    "dask_cuda>=24.2.0",
    "plotly>=5.20.0",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="AICS",
    author_email="benjamin.morris@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="general workflow for morphogenesis projects",
    entry_points={
        "console_scripts": [
            "run_morflowgenesis=morflowgenesis.bin.run_workflow:main",
        ],
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="morflowgenesis",
    name="morflowgenesis",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",
    setup_requires=setup_requirements,
    test_suite="morflowgenesis/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCell/morflowgenesis",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.3.0",
    zip_safe=False,
)
