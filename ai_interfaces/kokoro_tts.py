import soundfile as sf
from kokoro import KPipeline

def text_to_speech(text, voice='af_heart', output_file='output.wav', language_code='a'):
    # Initialize the Kokoro TTS pipeline
    pipeline = KPipeline(lang_code=language_code)

    # Generate speech from text
    generator = pipeline(text, voice=voice)

    # Save each segment of the generated audio
    for i, (gs, ps, audio) in enumerate(generator):
        segment_file = f'{i}_{output_file}'
        sf.write(segment_file, audio, 24000)
