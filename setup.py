#from setuptools import setup, find_packages
from pathlib import Path

def setup_dirs(dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

audio_dir = "tmp/audio/"
video_dir = "tmp/video/"
setup_dirs([audio_dir, video_dir])

# TODO this doesn't work, I still have to run manually
import nltk
nltk.download('averaged_perceptron_tagger_eng')

"""
setup(
    name="rooporter",
    version="0.1.0",
    url="https://github.com/RichSeibert/rooporter",
    packages=find_packages(where="HunyuanVideo") + find_packages(where="MeloTTS"),
    package_dir={
        "hyvideo": "HunyuanVideo/hyvideo",
        "melo": "MeloTTS/melo",
    },
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.10",
)
"""
