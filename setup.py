from setuptools import setup, find_packages
from pathlib import Path

def setup_dirs(dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

audio_dir = "tmp/audio/"
video_dir = "tmp/video/"
setup_dirs([audio_dir, video_dir])
import nltk
from nltk.data import path
nltk_data_dir = ".venv/nltk_data"
path.append(nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)


setup(
    name="rooporter",
    version="0.1.0",
    url="https://github.com/RichSeibert/rooporter",
    packages=find_packages(where="HunyuanVideo"),
    package_dir={"": "HunyuanVideo"},
    install_requires=[],
    python_requires=">=3.10",
)
