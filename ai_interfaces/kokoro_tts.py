import soundfile as sf
from kokoro import KPipeline

def text_to_speech(text, output_file, voice='af_heart', language_code='a'):
    pipeline = KPipeline(lang_code=language_code)
    generator = pipeline(text, voice=voice)
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(output_file, audio, 24000)
