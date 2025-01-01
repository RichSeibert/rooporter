from setuptools import setup, find_packages

setup(
    name="rooporter",
    version="0.1.0",
    url="https://github.com/RichSeibert/rooporter",
    packages=find_packages(where="HunyuanVideo"),
    package_dir={"": "HunyuanVideo"},
    install_requires=[],
    python_requires=">=3.10",
)
