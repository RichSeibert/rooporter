from setuptools import setup, find_packages
from pathlib import Path

def setup_dirs(dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

audio_dir = "tmp/audio/"
video_dir = "tmp/video/"
setup_dirs([audio_dir, video_dir])
import nltk
nltk.download('averaged_perceptron_tagger_eng', download_dir=".venv")


setup(
    name="rooporter",
    version="0.1.0",
    url="https://github.com/RichSeibert/rooporter",
    packages=find_packages(where="HunyuanVideo"),
    package_dir={"": "HunyuanVideo"},
    install_requires=[],
    python_requires=">=3.10",
)
