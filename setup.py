"""Setup configuration for pneumonia detection project."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            # Skip empty lines, comments, and -r includes
            if line and not line.startswith("#") and not line.startswith("-r"):
                requirements.append(line)
        return requirements

setup(
    name="pneumonia-detector",
    version="2.0.0",
    author="Pneumonia Detection Team",
    author_email="contact@pneumonia-detector.com",
    description="Modern CNN-based pneumonia detection from chest X-rays",
    long_description=read_readme() if os.path.exists("README.md") else "Modern pneumonia detection using deep learning",
    long_description_content_type="text/markdown",
    url="https://github.com/julihocc/Pneumonia-Detection-using-CNN",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt") if os.path.exists("requirements.txt") else [],
    extras_require={
        "dev": read_requirements("requirements-dev.txt") if os.path.exists("requirements-dev.txt") else [],
    },
    entry_points={
        "console_scripts": [
            "pneumonia-train=pneumonia_detector.cli:train_cli",
            "pneumonia-predict=pneumonia_detector.cli:predict_cli",
            "pneumonia-serve=pneumonia_detector.api:serve",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)