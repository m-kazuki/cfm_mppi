import os

import setuptools

NAME = "cfm_mppi"
DESCRIPTION = "Unified Generation-Refinement Planning: Bridging Guided Flow Matching and Sampling-Based MPC for Social Navigation"
EMAIL = "mizuta@uw.edu"
AUTHOR = "Kazuki Mizuta"
REQUIRES_PYTHON = ">=3.11.0"

readme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")

try:
    with open(readme_path) as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setuptools.setup(
    name=NAME,
    version="0.1",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=setuptools.find_packages(),
    extras_require={
        "dev": [
            "pre-commit",
            "black==22.6.0",
            "usort==1.0.4",
            "ufmt==2.3.0",
            "flake8==7.0.0",
            "pydoclint",
        ],
    },
    install_requires=["numpy", "torch", "torchdiffeq"],
    license="CC-by-NC",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
