import os

import numpy as np
import whisper
import time
import deepl
import openai
from TTS.api import TTS
import gradio as gr

# from IPython.display import Audio
# import IPython.display as ipd

wave_path = "./sound_inputs/"
stt_model = None
tts_model = None
translator = None


def main():
    prepare_models()

    gr.Interface(
        title="OpenAI Whisper ASR Gradio Web UI",
        fn=process_audio,
        inputs=[gr.Audio(source="microphone", type="filepath")],
        outputs=["audio"],
        live=True,
    ).launch()


def prepare_models():
    prepare_whisper_model()
    prepare_coqui_model()
    prepare_deepl_model()
    prepare_openai_model()


def prepare_whisper_model():
    global stt_model
    print("Prepare model")
    stt_model = whisper.load_model(name="base", download_root="./models/")
    print(stt_model.device)


def prepare_coqui_model():
    global tts_model
    model_name = "tts_models/en/vctk/vits"
    tts_model = TTS(model_name)
    print("Sample rate:", tts_model.synthesizer.output_sample_rate)


def prepare_deepl_model():
    global translator
    translator = deepl.Translator("5b403de4-54f7-052f-aa6f-ae202e557e05:fx")


def prepare_openai_model():
    openai.api_key = "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"


def process_audio(sound_input_path):
    # record audio
    sound_path = save_audio(sound_input_path)
    # transcribe audio
    transcript = transcribe_audio(sound_path)
    # remove audio file
    os.remove(sound_path)
    # translate the input to english
    transcript_translated = translate_transcript(transcript["text"])
    # text to speech english
    voice_arr = text_to_speech(transcript_translated)

    return voice_arr


def save_audio(path):
    os.makedirs(wave_path, exist_ok=True)
    target_path = os.path.join(wave_path, os.path.basename(path))
    os.rename(path, target_path)
    return target_path


def transcribe_audio(path):
    result = stt_model.transcribe(path)
    print(result)
    return result


def transcribe_audio_buffer(buffer):
    audio = whisper.pad_or_trim(buffer)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(stt_model.device)

    # detect the spoken language
    _, probs = stt_model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions(
        fp16=False if stt_model.device.type == "cpu" else True
    )
    result = whisper.decode(stt_model, mel, options)

    return result


def translate_transcript(transcript):
    result = translator.translate_text(
        transcript, source_lang="DE", target_lang="EN-US"
    )
    return result


def prompt_gpt(transcript):
    base_prompt = """The following is a conversation with a good friend called Lisa. Lisa works as a computer scientist and is creative, clever, very friendly and talks casually. Long answers are marked with [length=long], medium-length answers are marked with [length=medium] and short answers are marked with [length=short]. Also, the tone can be specified with [tone=friendly] and so on.

Human: Hi Lisa, what were we talking about yesterday? I forgot.
[length=long, tone=friendly+sarcastic] Lisa: You already forgot? Your memory is terrible haha. We were talking about this crazy idea you had for an app but you never told me what it was.
Human: Oh yeah, that's what I was trying to remember!
[length=short, tone=sarcastic] Lisa: Mustn't have been that good of an idea then...
Human: Nice as always... You still wanna hear about it?
[length=medium, tone=friendly] Lisa: Sure, I'm a little bit busy nowadays but this could be a fun little side project.
Human: """
    result = openai.Completion.create(
        model="text-davinci-003",
        prompt=base_prompt + transcript + "\n[length=medium, tone=friendly] Lisa:",
        max_tokens=100,
        temperature=0.9,
        stop="Human:",
        presence_penalty=0.1,
        frequency_penalty=0.3,
    )
    return result["choices"][0].text.strip()


def text_to_speech(text):
    global tts_model
    wav = tts_model.tts(text, speaker=tts_model.speakers[4])
    return (tts_model.synthesizer.output_sample_rate, np.array(wav))


def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    main()
