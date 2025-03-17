import sys
import os
# TODO this is shit
base_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, 'MeloTTS')))
from melo.api import TTS
from multiprocessing import set_start_method, Pool

def melo_tts_multithread(input_data, pool_size):
    logging.info("Generating audio")
    set_start_method("spawn", force=True)
    with Pool(processes=pool_size) as pool:
        pool.map(melo_tts, input_data)

def melo_tts(data):
    file_name, text = data
    speed = 1.25
    # WARN - meloTTS doesn't clean up gpu memmory. Using multiprocess fixes
    # this and adds the benefit of parallization
    device = 'auto' # Will automatically use GPU if available
    model = TTS(language='EN', device=device)
    speaker_ids = model.hps.data.spk2id
    output_path = f"tmp/audio/{file_name}.wav"
    model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

