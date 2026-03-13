"""Legacy setuptools configuration.

This project is configured primarily through pyproject.toml.
The setup.py file is kept for compatibility with older tooling.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file and return its content."""
    return path.read_text(encoding="utf-8")


def read_requirements(path: Path) -> list[str]:
    """Parse requirements file entries, skipping comments and blanks."""
    if not path.exists():
        return []
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            requirements.append(value)
    return requirements


setup(
    name="comp-cars-deep-learning-analysis",
    version="1.0.0",
    author="Diego Vega",
    author_email="diegovega2000@live.com",
    description="Vehicle analysis with deep learning using the CompCars dataset",
    long_description=read_text_file(ROOT / "README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/diegovega2001/Memoria",
    project_urls={
        "Homepage": "https://github.com/diegovega2001/Memoria",
        "Documentation": "https://github.com/diegovega2001/Memoria/blob/main/README.md",
        "Repository": "https://github.com/diegovega2001/Memoria.git",
        "Bug Tracker": "https://github.com/diegovega2001/Memoria/issues",
    },
    license="Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Chile",
    packages=find_packages(where=".", include=["src*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(ROOT / "requirements.txt"),
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
    include_package_data=True,
    zip_safe=False,
)
