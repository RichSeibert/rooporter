from pathlib import Path

def setup_dirs(dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

audio_dir = "tmp/audio/"
video_dir = "tmp/video/"
setup_dirs([audio_dir, video_dir])

import nltk
nltk.download('averaged_perceptron_tagger_eng')
