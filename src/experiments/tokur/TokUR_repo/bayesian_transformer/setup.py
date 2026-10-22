# setup.py
from setuptools import setup, find_packages

setup(
    name="bayesian_transformer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.19.2",
        "transformers>=4.10.0",
    ],
    author="Haizhou Shi @ Rutgers ML Lab",
    author_email="haizhou.shi.057@gmail.com",
    description="A simple package that converts any transformer into a Bayesian transformer in a Training-Free Manner (TFB).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/haizhou-shi/bayesian-lm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)