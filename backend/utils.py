from contextlib import contextmanager
import sys
import os
import io
from enum import Enum
import requests
import re
from datetime import datetime
import json
import wave

import numpy as np
import openai
import google.cloud.texttospeech as tts
import soundfile as sf
from thefuzz import fuzz

from eventlet import tpool

# import whisper

####################################################################
#####     DATA STRUCTS      ########################################
####################################################################


class Mode:
    DORMANT = "DORMANT"
    SYSTEM = "SYSTEM"
    CONVERSATION = "CONVERSATION"
    TRANSLATION = "TRANSLATION"
    INFORMATION = "INFORMATION"


class Language:
    GERMAN = "DE"
    ENGLISH = "EN-US"

    @staticmethod
    def get(name):
        if name == "DE":
            return "DE"
        if name == "EN-US":
            return "EN-US"
        raise Exception(f"Language {name} not yet supported!")


class Gender:
    FEMALE = "female"
    MALE = "male"

    @staticmethod
    def get(gender):
        if gender == "female":
            return "female"
        if gender == "male":
            return "male"
        raise Exception(f"Gender {gender} not yet supported!")


class Mood:
    NEUTRAL = 0
    HAPPY = 1
    FEISTY = 2
    SAD = 3
    MAD = 4

    # leaving this in cause typing that out was tedious
    MOOD_TRANSITION = [
        [0.8, 0.12, 0.05, 0.02, 0.01],
        [0.5, 0.45, 0.05, 0.0, 0.0],
        [0.8, 0.1, 0.1, 0.0, 0.0],
        [0.7, 0.0, 0.0, 0.15, 0.05],
        [0.8, 0.0, 0.0, 0.05, 0.15],
    ]

    def __init__(self, style_en="a witty sarcastic friend", logger=None):
        self.mood = Mood.NEUTRAL
        self.style = {}
        self.style[Language.ENGLISH] = style_en
        self.style_example = {}
        self.style_example[Language.ENGLISH] = self._get_style_example(style_en, logger)
        self.logger = logger
        logger.log(
            Mode.SYSTEM,
            "system",
            f"Style example: {self.style_example[Language.ENGLISH]}",
        )

    def _get_style_example(self, style_en, logger):
        style_prompt = [
            {"role": "system", "content": "You are a speech information system"},
            {
                "role": "user",
                "content": f'How would you write a stereotypical response with the style "{style_en}" to the sentence "Hey, wanna hang out?" in text? If you cannot due to your guidelines then give a response that is still within your guidelines instead. Answer with exactly one sentence in quotation marks.',
            },
        ]
        for i in range(5):
            try:
                result = openai.ChatCompletion.create(
                    model=gpt_models[Mode.CONVERSATION],
                    messages=style_prompt,
                    max_tokens=250,
                    temperature=0.7,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
            except Exception:
                self.style[Language.ENGLISH] = "regular person"
                return "Sure, do you have anything in mind?"

            logger.track_stats(api="chatgpt", tokens=result["usage"]["total_tokens"])
            answer = result["choices"][0]["message"]["content"]
            if (
                sum(
                    (
                        "AI language model" in answer,
                        "I cannot" in answer,
                        "stereotypical" in answer,
                        "ethical" in answer,
                        "guidelines" in answer,
                        "against" in answer,
                        "provide" in answer,
                    )
                )
                < 2
            ):
                break
            else:
                if i == 0:
                    style_prompt.append(
                        {
                            "role": "assistant",
                            "content": f'I will provide you with responses that gradually reach the style "{style_en}", beginning with 1 as a neutral answer and going until 3 which is my limit.',
                        }
                    )
                    style_prompt.append(
                        {
                            "role": "assistant",
                            "content": f"1. Sounds good. What do you have in mind?",
                        }
                    )
                print(
                    f"DEBUG retry style example prompt after {answer} with new style_prompt",
                    style_prompt,
                )
        return re.sub(r"[^a-zA-Z0-9.,!? ]", "", re.sub(r"^[0-9]\.\s*", "", answer))

    def translate_style(self, translation_model, language):
        self.style[language] = translate_transcript(
            translation_model, self.style[Language.ENGLISH], Language.ENGLISH, language
        ).text
        self.logger.track_stats("deepl", self.style[Language.ENGLISH])
        self.style_example[language] = translate_transcript(
            translation_model,
            self.style_example[Language.ENGLISH],
            Language.ENGLISH,
            language,
        ).text
        self.logger.track_stats("deepl", self.style_example[Language.ENGLISH])

    def get_message_length(self, user_msg):
        length_options = [
            "very short",
            "short",
            "medium",
            "long",
            "very long",
            "long paragraph",
        ]
        length_probabilities = [0.05, 0.20, 0.40, 0.25, 0.05, 0.05]

        if np.random.random() < 0.75:
            return length_options[
                np.random.choice(len(length_options), p=length_probabilities)
            ]
        else:
            msg_length = len(user_msg)
            if msg_length < 20:
                return "very short"
            elif msg_length < 40:
                return "short"
            elif msg_length < 80:
                return "medium"
            elif msg_length < 200:
                return "long"
            elif msg_length < 500:
                return "very long"
            else:
                return "long paragraph"


class MessagePair:
    def __init__(
        self, msg_user=None, msg_charlie_dict=None, reply_style=None, reply_length=None
    ):
        self.msg_user = msg_user
        self.msg_charlie_raw = None
        self.msg_charlie_clean = None
        self.msg_charlie_style = None
        if "raw" in msg_charlie_dict:
            self.msg_charlie_raw = msg_charlie_dict["raw"]
        if "clean" in msg_charlie_dict:
            self.msg_charlie_clean = msg_charlie_dict["clean"]
        if "style" in msg_charlie_dict:
            self.msg_charlie_style = msg_charlie_dict["style"]

        # reply style that was used to generate the response of chat gpt
        self.reply_style = reply_style
        self.reply_length = reply_length


class RecordingState:
    is_recording = False
    recorded_frames = 0


class RecordingParams:
    def __init__(self, config):
        self.buffer_length_seconds = config["recording_settings"][
            "buffer_length_seconds"
        ]
        self.sample_rate = config["recording_settings"]["sample_rate"]
        self.chunk_size = config["recording_settings"]["chunk_size"]
        self.activation_threshold = config["recording_settings"]["activation_threshold"]
        self.deactivation_threshold = config["recording_settings"][
            "deactivation_threshold"
        ]
        self.activation_window = config["recording_settings"]["activation_window"]
        self.deactivation_window = config["recording_settings"]["deactivation_window"]


class AudioProcessor:
    def __init__(
        self,
        buffer_length=30,
        max_recording_length=30,
        sample_rate=16000,
        chunk_size=16000,
        chunk_step=1600,
        activation_threshold=550,
        deactivation_threshold=250,
        activation_chunk_threshold=1,
        deactivation_chunk_threshold=10,
        reset_chunk_threshold=2,
        logger=None,
    ):
        assert max_recording_length <= buffer_length

        self.buffer_length = buffer_length
        self.max_recording_length = max_recording_length
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.chunk_step = chunk_step
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.activation_chunk_threshold = activation_chunk_threshold
        self.deactivation_chunk_threshold = deactivation_chunk_threshold
        self.reset_chunk_threshold = reset_chunk_threshold
        self.logger = logger

        self.trigger_chunks = 0
        self.reset_chunk_counter = 0
        self.buffer = np.zeros(buffer_length * sample_rate, dtype=np.int16)
        self.buffer_current_index = 0
        self.buffer_chunk_index = 0
        self.recording_start_index = 0
        self.recording_duration = 0
        self.recording_state = "LISTENING"
        self.recording_buffer = np.zeros(1, dtype=np.int16)

    def append_and_process(self, buffer_arr):
        # if append buffer exceeds buffer size, split and append
        if self.buffer_current_index + buffer_arr.shape[0] > self.buffer.shape[0]:
            rollover = True
            frames_leftover = buffer_arr.shape[0] - (
                self.buffer.shape[0] - self.buffer_current_index
            )
            self.buffer[self.buffer_current_index :] = buffer_arr[:-frames_leftover]
            self.buffer[:frames_leftover] = buffer_arr[
                buffer_arr.shape[0] - frames_leftover :
            ]
            self.buffer_current_index = frames_leftover
        else:
            rollover = False
            self.buffer[
                self.buffer_current_index : self.buffer_current_index
                + buffer_arr.shape[0]
            ] = buffer_arr
            self.buffer_current_index += buffer_arr.shape[0]

        # process new chunks
        events = []
        actual_index = (
            self.buffer_current_index
            if not rollover
            else self.buffer_current_index + self.buffer.shape[0]
        )
        frames_to_process = actual_index - self.buffer_chunk_index
        while frames_to_process > self.chunk_step:
            chunk = np.zeros(self.chunk_size)
            if self.buffer_chunk_index + self.chunk_size > self.buffer.shape[0]:
                frames_leftover = (
                    self.buffer_chunk_index + self.chunk_size - self.buffer.shape[0]
                )
                chunk[:-frames_leftover] = self.buffer[self.buffer_chunk_index :]
                chunk[self.chunk_size - frames_leftover :] = self.buffer[
                    :frames_leftover
                ]
            else:
                chunk[:] = self.buffer[
                    self.buffer_chunk_index : self.buffer_chunk_index + self.chunk_size
                ]

            if self.recording_state == "LISTENING":
                if np.abs(chunk).mean() > self.activation_threshold:
                    self.trigger_chunks += 1
                else:
                    self.reset_chunk_counter += 1
                    if self.reset_chunk_counter >= self.reset_chunk_threshold:
                        self.trigger_chunks = 0
                        self.reset_chunk_counter = 0
                if self.trigger_chunks >= self.activation_chunk_threshold:
                    self.recording_state = "RECORDING"
                    self.trigger_chunks = 0
                    self.reset_chunk_counter = 0
                    # rollback to not miss the start of the sentence
                    self.recording_start_index = self.buffer_chunk_index - int(
                        1.2 * self.chunk_size
                    )
                    if self.recording_start_index < 0:
                        self.recording_start_index += self.buffer.shape[0]
                    self.recording_duration = int(1.2 * self.chunk_size)
                    events.append("START")
                    if self.logger:
                        self.logger.log(Mode.SYSTEM, "system", "Start recording..")
            else:
                self.recording_duration += self.chunk_step
                if np.abs(chunk).mean() < self.deactivation_threshold:
                    self.trigger_chunks += 1
                else:
                    self.reset_chunk_counter += 1
                    if self.reset_chunk_counter >= self.reset_chunk_threshold:
                        self.trigger_chunks = 0
                        self.reset_chunk_counter = 0
                if self.trigger_chunks >= self.deactivation_chunk_threshold:
                    self.recording_state = "LISTENING"
                    self.trigger_chunks = 0
                    self.reset_chunk_counter = 0
                    events.append("STOP")
                    if (
                        self.recording_start_index + self.recording_duration
                        > self.buffer.shape[0]
                    ):
                        frames_leftover = (
                            self.recording_start_index
                            + self.recording_duration
                            - self.buffer.shape[0]
                        )
                        self.recording_buffer = np.zeros(
                            self.recording_duration, dtype=np.int16
                        )
                        self.recording_buffer[
                            : self.recording_duration - frames_leftover
                        ] = self.buffer[self.recording_start_index :]
                        self.recording_buffer[
                            self.recording_duration - frames_leftover :
                        ] = self.buffer[:frames_leftover]
                    else:
                        self.recording_buffer = np.zeros(
                            self.recording_duration, dtype=np.int16
                        )
                        self.recording_buffer[:] = self.buffer[
                            self.recording_start_index : self.recording_start_index
                            + self.recording_duration
                        ]
                    if self.logger:
                        self.logger.log(
                            Mode.SYSTEM,
                            "system",
                            f"Deactivate recording, recorded frames: {self.recording_duration} = {self.recording_duration / self.sample_rate} seconds",
                        )

            self.buffer_chunk_index = (
                self.buffer_chunk_index + self.chunk_step
            ) % self.buffer.shape[0]
            frames_to_process -= self.chunk_step
        if len(events) == 0:
            return None
        if events[0] == "START":
            return "START"
        elif events[0] == "STOP":
            self.buffer[:] = 0
            return "STOP"
        else:
            return "STARTSTOP"

    def save_recording_to_file(self, path):
        with open(path, "wb") as wavfile:
            print("DEBUG recording buffer shape", self.recording_buffer.shape)
            sf.write(wavfile, self.recording_buffer, self.sample_rate, format="WAV")


gpt_models = {
    # Mode.CONVERSATION: "text-davinci-003",
    Mode.CONVERSATION: "gpt-3.5-turbo",
    Mode.INFORMATION: "gpt-3.5-turbo",
}

openai.api_key = json.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.json"), "r")
)["api_keys"]["openai"]

####################################################################
#####     LOGGING      #############################################
####################################################################


class Logger:
    def __init__(self, session_token, userUID, socketio=None):
        self.session_token = session_token
        self.logfile_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logfiles", userUID
        )
        self.stats_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "stats"
        )
        os.makedirs(
            self.logfile_dir,
            exist_ok=True,
        )
        os.makedirs(self.stats_dir, exist_ok=True)
        self.session_time = datetime.now()
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.filename = os.path.join(
            self.logfile_dir, f"session_{timestamp}.txt".replace(":", "-")
        )
        self.last_log_message = f"[{timestamp}][SYSTEM, system] Start session."
        print(self.last_log_message)
        with open(self.filename, "w") as logfile:
            logfile.write(self.last_log_message + "\n")

        self.socketio = socketio
        if socketio is not None:
            print("emit conversationmessage", self.session_token, self.last_log_message)
            self.socketio.emit(
                "convmsg",
                {"session_token": self.session_token, "message": self.last_log_message},
            )

        self.stats = self._init_stats()

    def _init_stats(self):
        today_date = datetime.now()
        self.stats_file_path = os.path.join(
            self.stats_dir, f"stats_{today_date.year}-{today_date.month}.json"
        )
        if os.path.isfile(self.stats_file_path):
            with open(self.stats_file_path, "r") as stats_file:
                return json.load(stats_file)
        return {
            # usage info
            "whisper_minutes": 0,
            "chatgpt_tokens": 0,
            "deepl_character": 0,
            "google_tts_character": 0,
            "elevenlabs_tts_character": 0,
            # pricing info
            "whisper_cost": 0,
            "chatgpt_cost": 0,
            "deepl_percentage_used": 0,
            "google_tts_percentage_used": 0,
            "google_tts_cost": 0,
            "elevenlabs_tts_percentage_used": 0,
        }

    def log(self, mode, name, text, verbose=True):
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.last_log_message = f"[{timestamp}][{mode}, {name}] {text}"
        if verbose:
            print(self.last_log_message)
        with open(self.filename, "a") as logfile:
            logfile.write(self.last_log_message + "\n")

        if self.socketio is not None:
            print("emit conversationmessage", self.session_token, self.last_log_message)
            self.socketio.start_background_task(
                self.socketio.emit,
                "convmsg",
                {"session_token": self.session_token, "message": self.last_log_message},
            )

    def get_last_log_message(self):
        return self.last_log_message

    def print_log(self):
        with open(self.filename, "r") as logfile:
            for line in logfile:
                print(line)

    def track_stats(self, api, message="", duration=0.0, tokens=0):
        if api == "whisper":
            self.stats["whisper_minutes"] += duration / 60.0
            self.stats["whisper_cost"] = self.stats["whisper_minutes"] * 0.006
        elif api == "chatgpt":
            self.stats["chatgpt_tokens"] += tokens
            self.stats["chatgpt_cost"] = self.stats["chatgpt_tokens"] * 0.002 / 1000.0
        elif api == "deepl":
            self.stats["deepl_character"] += len(message)
            self.stats["deepl_percentage_used"] = (
                self.stats["deepl_character"] / 500000.0
            )
        elif api == "google_tts":
            self.stats["google_tts_character"] += len(message)
            self.stats["google_tts_percentage_used"] = (
                self.stats["google_tts_character"] / 1000000.0
            )
            self.stats["google_tts_cost"] = (
                np.max([self.stats["google_tts_character"] - 1000000, 0])
            ) * 0.000016
        elif api == "elevenlabs_tts":
            api_status = _elevenlabs_tts_api_ready(Language.ENGLISH, "")
            if not api_status["api_ready"]:
                return
            self.stats["elevenlabs_tts_character"] = api_status["character_count"]
            self.stats["elevenlabs_tts_limit"] = api_status["character_limit"]
            self.stats["elevenlabs_tts_percentage_used"] = (
                api_status["character_count"] / api_status["character_limit"]
            )
        else:
            raise Exception(f"API {api} is not a valid API to track!")
        with open(self.stats_file_path, "w") as stats_file:
            json.dump(self.stats, stats_file, indent=4)


####################################################################
#####     MODELS      ##############################################
####################################################################


def get_google_tts_voice_name_and_settings(language, gender):
    voice_name_dict = {
        Language.ENGLISH: {
            Gender.MALE: ["en-US-Neural2-D", 1.05, -1.0],
            Gender.FEMALE: ["en-US-Neural2-H", 1.05, -2.0],
        },
        Language.GERMAN: {
            Gender.MALE: ["de-DE-Neural2-B", 1.1, 0.0],
            Gender.FEMALE: ["de-DE-Neural2-C", 1.1, -4.0],
        },
    }
    return voice_name_dict[language][gender]


def get_tts_model_name(language):
    if language == Language.GERMAN:
        return "tts_models/de/thorsten/vits"
    elif language == Language.ENGLISH:
        return "tts_models/en/vctk/vits"
    return "tts_models/en/vctk/vits"


def transcribe_audio_buffer_api(buffer_arr, sample_rate):
    memory_file = io.BytesIO()
    memory_file.name = "transcript.wav"
    sf.write(memory_file, buffer_arr, sample_rate, format="WAV")
    memory_file.seek(0)

    transcription = openai.Audio.transcribe("whisper-1", memory_file)
    return transcription["text"]


def translate_transcript(translation_model, transcript, source_lang, target_lang):
    if source_lang == "EN-US":
        source_lang = "EN"
    result = translation_model.translate_text(
        transcript, source_lang=source_lang, target_lang=target_lang
    )
    return result


def text_to_speech(tts_model, text, gender):
    if tts_model.speakers is not None:
        if gender == "male":
            # 10, 26, 29, 32, 41
            target_speaker = tts_model.speakers[min(len(tts_model.speakers), 29)]
        else:
            # 3, 15, 17, 19, 22, 23, 37, 44, 47
            target_speaker = tts_model.speakers[min(len(tts_model.speakers), 22)]
        wav = tts_model.tts(text, speaker=target_speaker)
    else:
        wav = tts_model.tts(text)

    play_output_audio(tts_model.synthesizer.output_sample_rate, np.array(wav))


def text_to_speech_api(
    tts_method, language, gender, text, socketio=None, session_token="", logger=None
):
    if tts_method == "google_tts":
        voice_name_settings = get_google_tts_voice_name_and_settings(language, gender)
        language_code = "-".join(voice_name_settings[0].split("-")[:2])
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code, name=voice_name_settings[0]
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            speaking_rate=voice_name_settings[1],
            pitch=voice_name_settings[2],
            effects_profile_id=["large-home-entertainment-class-device"],
        )
        try:
            client = tts.TextToSpeechClient()
        except:
            if logger is not None:
                socketio.start_background_task(
                    logger.log,
                    Mode.SYSTEM,
                    "system",
                    f"Could not establish Google cloud TTS, check your CHARLIE config files and Google cloud api settings",
                )
            return
        response = client.synthesize_speech(
            input=text_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        audio_content = response.audio_content
    elif tts_method == "elevenlabs_tts":
        api_status = _elevenlabs_tts_api_ready(language, text)
        if not api_status["api_ready"]:
            if logger is not None:
                socketio.start_background_task(
                    logger.log,
                    Mode.SYSTEM,
                    "system",
                    f"elevenlabs.ai API not ready either cause wrong settings or no quota left",
                )
            return
        voice_id = (
            "yoZ06aMxZJJ28mfd3POQ" if gender == Gender.MALE else "AZnzlk1XvdvUeBnXmlld"
        )
        res = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": os.environ["ELEVENLABS_TTS_CREDENTIALS"]},
            json={
                "text": text,
                "voice_settings": {"stability": 0, "similarity_boost": 0},
            },
        )
        if not res.ok:
            return
        audio_content = res.content
    else:
        return
    if socketio is None:
        import simpleaudio as sa

        with io.BytesIO(audio_content) as ac:
            wavf = wave.open(ac)
            sr = wavf.getframerate()
            play_obj = sa.play_buffer(audio_content, 1, 2, sr)
            # wait for playback to finish before exiting
            play_obj.wait_done()
            wavf.close()
    else:
        socketio.start_background_task(
            socketio.emit,
            "responseaudio",
            {
                "session_token": session_token,
                "audio_content": audio_content,
            },
        )
        output_sound_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "..",
            "live2d",
            "Samples",
            "Resources",
            "output_sounds",
            session_token,
        )
        os.makedirs(output_sound_dir, exist_ok=True)
        sf_content = sf.SoundFile(io.BytesIO(audio_content))
        with open(
            os.path.join(
                output_sound_dir,
                "output_sound.wav",
            ),
            "wb",
        ) as outfile:
            # outfile.write(audio_content)
            sf.write(outfile, sf_content.read(), sf_content.samplerate)
        socketio.start_background_task(socketio.emit, "live2dlipsync", session_token)


def _elevenlabs_tts_api_ready(language, text):
    status = requests.get(
        "https://api.elevenlabs.io/v1/user/subscription",
        headers={"xi-api-key": os.environ["ELEVENLABS_TTS_CREDENTIALS"]},
    )
    ready_info = {
        "api_status": status.ok,
        "api_ready": False,
        "character_count": None,
        "character_limit": None,
        "character_leftover": None,
    }
    if not status.ok:
        return ready_info
    ready_info["api_ready"] = all(
        (
            language == Language.ENGLISH,
            "character_limit" in status.json(),
            "character_count" in status.json(),
            (
                "character_limit" in status.json()
                and "character_count" in status.json()
                and len(text)
                < status.json()["character_limit"] - status.json()["character_count"]
            ),
        )
    )

    if ready_info["api_ready"]:
        ready_info["character_count"] = status.json()["character_count"]
        ready_info["character_limit"] = status.json()["character_limit"]
        ready_info["character_leftvoer"] = (
            status.json()["character_limit"] - status.json()["character_count"]
        )
        return ready_info
    else:
        return ready_info


def prompt_gpt(
    mode,
    text,
    language,
    name,
    translation_model,
    memory_buffer=None,
    remembered_message_count=3,
    mood=None,
    logger=None,
):
    if mode == Mode.CONVERSATION:
        chat_gpt_messages, reply_style, reply_length = get_conversation_prompt_chat_gpt(
            translation_model,
            text,
            language,
            name,
            memory_buffer,
            remembered_message_count,
            mood,
        )
        try:
            result = openai.ChatCompletion.create(
                model=gpt_models[mode],
                messages=chat_gpt_messages,
                max_tokens=250,
                temperature=0.3,
                presence_penalty=0.67,
                frequency_penalty=1.03,
            )
        except openai.error.RateLimitError as e:
            err_msg = str(e)
            if "exceeded your current quota" in err_msg:
                ret_msg = "I think our time together came to an end. I have to say goodbye now."
                return (
                    {"raw": ret_msg, "clean": ret_msg, "style": ret_msg},
                    "rate quota exception answer style",
                )
            else:
                ret_msg = "Wow you're talking so fast, let me think about that for a moment..."
                return (
                    {"raw": ret_msg, "clean": ret_msg, "style": ret_msg},
                    "rate exception answer style",
                )
        except Exception as e:
            err_msg = str(e)
            ret_msg = (
                "I think something went terribly wrong here. We need an adult in here."
            )
            return (
                {"raw": ret_msg, "clean": ret_msg, "style": ret_msg},
                "general exception answer style",
            )
        logger.track_stats(api="chatgpt", tokens=result["usage"]["total_tokens"])
        return (
            extract_prompt_answers(result["choices"][0]["message"]["content"]),
            reply_style,
            reply_length,
        )
    elif mode == Mode.INFORMATION:
        base_prompt = get_base_prompt("information", language)
        base_prompt.append({"role": "user", "content": text})
        try:
            result = openai.ChatCompletion.create(
                model=gpt_models[mode],
                messages=base_prompt,
                max_tokens=150,
                temperature=0.5,
                presence_penalty=0.3,
                frequency_penalty=0.3,
            )

        except openai.error.RateLimitError as e:
            err_msg = str(e)
            return f"RateLimitError with the message {err_msg}", None
        except Exception as e:
            err_msg = str(e)
            return f"Error with the message {err_msg}", None
        logger.track_stats(api="chatgpt", tokens=result["usage"]["total_tokens"])
        return result["choices"][0]["message"]["content"], None
    else:
        if language != Language.ENGLISH:
            logger.track_stats(
                api="deepl", message="There was an error with the GPT API"
            )
            return (
                translation_model.translate_text(
                    "There was an error with the GPT API",
                    source_lang="EN",
                    target_lang=language,
                ).text,
                None,
            )
        else:
            return "There was an error with the GPT API", None


def prompt_gpt_settings(input_text, language):
    base_prompt = get_base_prompt("settings", language)
    base_prompt.append({"role": "user", "content": input_text})

    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=base_prompt,
        max_tokens=100,
        temperature=0.13,
        presence_penalty=0,
        frequency_penalty=0,
    )
    settings_string = result["choices"][0]["message"]["content"]

    settings_list = (
        settings_string.lower()
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .replace("'", "")
        .replace('"', "")
        .split(",")
    )
    formatted_settings_list = []
    if settings_list[0] == "conversation":
        formatted_settings_list.append(Mode.CONVERSATION)
    elif settings_list[0] == "translation":
        formatted_settings_list.append(Mode.TRANSLATION)
    elif settings_list[0] == "information":
        formatted_settings_list.append(Mode.INFORMATION)
    else:
        formatted_settings_list.append(None)

    if settings_list[1].lower() == "none":
        formatted_settings_list.append(None)
    else:
        formatted_settings_list.append(settings_list[1])

    if settings_list[2] == "german":
        formatted_settings_list.append(Language.GERMAN)
    elif settings_list[2] == "english":
        formatted_settings_list.append(Language.ENGLISH)
    else:
        formatted_settings_list.append(None)

    if settings_list[3] == "german":
        formatted_settings_list.append(Language.GERMAN)
    elif settings_list[3] == "english":
        formatted_settings_list.append(Language.ENGLISH)
    else:
        formatted_settings_list.append(None)

    return formatted_settings_list


####################################################################
#####     MISC      ################################################
####################################################################


def simple_match_shutdown(input_text):
    text = re.sub("\s+", "", input_text.lower())
    kws = [
        "charlieherunterfahren",
        "charlieshutdown",
        "charlyherunterfahren",
        "charlyshutdown",
    ]
    scores = []
    for kw in kws:
        scores.append(fuzz.ratio(text, kw))
        scores.append(fuzz.partial_ratio(text, kw))
        scores.append(fuzz.token_sort_ratio(text, kw))
        scores.append(fuzz.token_set_ratio(text, kw))
    return max(scores) >= 80


def simple_match_list_settings(input_text):
    text = re.sub("\s+", "", input_text.lower())
    kws = [
        "charliesettingslistallsettings",
        "charlieeinstellungenzeigealleeinstellungen",
        "charlysettingslistallsettings",
        "charlyeinstellungenzeigealleeinstellungen",
    ]
    scores = []
    for kw in kws:
        scores.append(fuzz.ratio(text, kw))
        scores.append(fuzz.partial_ratio(text, kw))
        scores.append(fuzz.token_sort_ratio(text, kw))
        scores.append(fuzz.token_set_ratio(text, kw))
    return max(scores) >= 80


def simple_match_change_settings(input_text):
    text = re.sub("\s+", "", input_text.lower())[:22]
    kws = [
        "charlieeinstellungen",
        "charliesettings",
        "charlieseinstellungen",
        "charliessettings",
        "charlyeinstellungen",
        "charlysettings",
        "charlyseinstellungen",
        "charlyssetings",
    ]
    scores = []
    for kw in kws:
        scores.append(fuzz.ratio(text, kw))
        scores.append(fuzz.partial_ratio(text, kw))
        scores.append(fuzz.token_sort_ratio(text, kw))
        scores.append(fuzz.token_set_ratio(text, kw))
    return max(scores) >= 80


def clean_settings_input(input_text):
    return re.sub("^[^a-zA-Z0-9]+", "", input_text)


def play_output_audio(sample_rate, voice_arr):
    import simpleaudio as sa

    audio = voice_arr * 32767.0 / np.max(np.abs(voice_arr))
    # convert to 16-bit data
    audio = audio.astype(np.int16)
    # start playback
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    # wait for playback to finish before exiting
    play_obj.wait_done()


def buffer_to_numpy(buffer, recorded_frames):
    if recorded_frames > len(buffer):
        relevant_frames = len(buffer)
    else:
        relevant_frames = recorded_frames
    return np.array(buffer)[-relevant_frames:].astype(np.float32) / 32768.0


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


####################################################################
#####     PROMPTS & MESSAGES     ###################################
####################################################################


def get_settings_result_string(translation_model, settings_type, setting, language):
    settings_results_string = ""
    if settings_type == "language_target":
        settings_results_string += f"Change translation target language to {setting}"
    elif settings_type == "language_source":
        settings_results_string += f"Change translation source language to {setting}"
    elif settings_type == "language":
        settings_results_string += f"Change Charlie's language to {setting}"
    elif settings_type == "gender":
        settings_results_string += f"Change Charlie's gender to {setting}"
    elif settings_type == "mode":
        settings_results_string += f"Change mode to {setting}"
    if language != Language.ENGLISH:
        return translation_model.translate_text(
            settings_results_string, source_lang="EN", target_lang=language
        ).text
    else:
        return settings_results_string


def get_error_message(translation_model, language):
    if language != Language.ENGLISH:
        return translation_model.translate_text(
            "Something went wrong. Please try again.",
            source_lang="EN",
            target_lang=language,
        ).text
    else:
        return "There was an error with the GPT API"


def get_base_prompt(prompt_type, language, name="John"):
    if prompt_type == "settings":
        return get_settings_prompt(language)
    elif prompt_type == "information":
        return get_information_prompt(language)
    elif prompt_type == "conversation":
        assert False
    assert False


# translate prompts or keep handcrafted?
def get_information_prompt(language):
    if language == Language.GERMAN:
        return [
            {
                "role": "system",
                "content": "Du bist ein hilfreicher KI Assistent und antwortest kurz und präzise auf Fragen.",
            }
        ]
    elif language == Language.ENGLISH:
        return [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that provides concise and precise answers to questions.",
            }
        ]
    else:
        assert False


# translate prompts or keep handcrafted?
def get_settings_prompt(language):
    if language == Language.GERMAN:
        return [
            {
                "role": "system",
                "content": "Du bist ein Kontrolleur welcher Parameter aus einem gegebenen Input extrahiert und auf Englisch ausgibt.",
            },
            {
                "role": "user",
                "content": 'Extrahiere den modus (unterhaltung, übersetzung, information), das gewünschte Geschlecht, die Sprache und die Ziel-Sprache aus dem folgenden Input. Falls du einzelne parameter nicht bestimmen kannst, gebe für denjenigen Parameter ein "None" aus. Übersetze die parameter auf Englisch und gebe sie als python array aus in der Form [modus, geschlecht, sprache, ziel-sprache]. Verändere den "modus" Parameter nur wenn das Wort "modus" im user input existiert, gebe ansonsten ein "None" für den "modus" aus.',
            },
            {
                "role": "assistant",
                "content": "Okay, ich habe verstanden. Ich werde mich daran erinnern nur den 'modus' zu ändern wenn der User das Wort 'modus' erwähnt.",
            },
            {
                "role": "user",
                "content": "Charlie einstellungen wechsel zum Übersetzungsmodus von Deutsch auf Englisch.",
            },
            {
                "role": "assistant",
                "content": "['translation', 'None', 'German', 'English']",
            },
            {
                "role": "user",
                "content": "Charlie einstellungen Bitte ändere das Geschlecht zu weiblich und die Sprache zu Deutsch.",
            },
            {"role": "assistant", "content": "['None', 'female', 'German', 'None']"},
            {
                "role": "user",
                "content": "Charlie einstellungen ändere das Geschlecht zu männlich und die Zielsprache zu Deutsch.",
            },
            {"role": "assistant", "content": "['None', 'male', 'None', 'German']"},
            {
                "role": "user",
                "content": "Charlie einstellungen wechselt die Sprache zu Englisch.",
            },
            {"role": "assistant", "content": "['None', 'None', 'English', 'None']"},
        ]
    elif language == Language.ENGLISH:
        return [
            {
                "role": "system",
                "content": "You are a controller that extracts parameters from a given input.",
            },
            {
                "role": "user",
                "content": "Extract the mode (conversation, translation, information), the desired gender, the language and the target-language from the following user input. If you can't determine individual parameters, then write \"None\" for the one you couldn't determine. Put the parameters in a python style array of the form [mode, gender, language, target-language]. Only change the 'mode' parameter if there exist the word 'mode' in the user input, otherwise output 'None' for the 'mode'.",
            },
            {
                "role": "assistant",
                "content": "Okay, I understood. I will remember to only change the 'mode' if the user inputs the word 'mode'.",
            },
            {
                "role": "user",
                "content": "Charlie settings switch to translation mode from German to English.",
            },
            {
                "role": "assistant",
                "content": "['translation', 'None', 'German', 'English']",
            },
            {
                "role": "user",
                "content": "Charlie settings please change the gender to female and the language to German.",
            },
            {"role": "assistant", "content": "['None', 'female', 'German', 'None']"},
            {
                "role": "user",
                "content": "Charlie settings change the gender to male and the target language to German.",
            },
            {"role": "assistant", "content": "['None', 'male', 'None', 'German']"},
            {
                "role": "user",
                "content": "Charlie settings change language to German.",
            },
            {"role": "assistant", "content": "['None', 'None', 'German', 'None']"},
        ]
    else:
        assert False


def get_conversation_prompt_chat_gpt(
    translation_model,
    text,
    language,
    name,
    memory_buffer,
    remembered_message_count,
    mood,
):
    print("DEBUG MEMORY", remembered_message_count, len(memory_buffer))

    # select the last N messages based on memory buffer size and configuration
    relevant_message_pairs = []
    if remembered_message_count > len(memory_buffer):
        remembered_message_count = len(memory_buffer)
    for i in range(remembered_message_count):
        relevant_message_pairs.append(memory_buffer[-1 - i])

    prompt = []
    print(
        "DEBUG",
        relevant_message_pairs,
        text,
        language,
        name,
        memory_buffer,
        remembered_message_count,
    )

    if language not in mood.style or language not in mood.style_example:
        mood.translate_style(translation_model, language)
    mood_style = mood.style[language]
    style_example = mood.style_example[language]
    message_length = mood.get_message_length(text)

    if language == Language.ENGLISH:
        prompt += [
            {
                "role": "system",
                "content": "You are a dialog and style completion engine.",
            },
            {
                "role": "user",
                "content": f'The following is a meta description of a dialog. I will provide you with a dialog excerpt and you fill in the message inside the <msg style="..." length="..."></msg> tag according to the "style" and "length" parameters (length="short" means a few words, length="very long" means at least 3 to 4 long sentences or even a paragraph). Emphasize the character traits. You never ask how you can help {name} or what you can do for {name}. Here is an example:\nI give you the input:\n{name}: Yo, wanna hang out?\nCharlie: <msg style="{mood_style}" length="short"></msg>\n\nAnd you give me the output without any extra information, i.e. follow this format and be concise and never ask how you can help {name} or what you can do for {name}:\nCharlie: {style_example}\n\nDo you understand?',
            },
            {
                "role": "assistant",
                "content": f'Yes, I understand. Please provide me with the dialog excerpt and the desired <msg style="..." length="..."></msg> tag for me to fill in.',
            },
        ]
    elif language == Language.GERMAN:
        prompt += [
            {
                "role": "system",
                "content": "Du bist eine Engine zur Vervollständigung von Dialogen und Stilen.",
            },
            {
                "role": "user",
                "content": f'Das Folgende ist eine Meta-Beschreibung eines Dialogs. Ich stelle dir einen Dialogauszug zur Verfügung, und du füllst die Nachricht innerhalb des Tags <msg style="..." length="..."></msg> entsprechend den "style"- und "length"-Parametern aus (length="short" bedeutet ein paar Wörter, length="very long" bedeutet mindestens 3 bis 4 lange Sätze oder sogar einen Absatz). Hebe die Charaktereigenschaften hervor. Du fragst nie, wie du {name} helfen könntest oder was du für {name} tun könntest. Hier ist ein Beispiel:\nIch gebe dir den Input:\n{name}: Hey, willst du abhängen?\nCharlie: <msg style="{style_example}" length="short"></msg>\n\nUnd du gibst mir die Antwort ohne zusätzliche Informationen, d.h. folge diesem Format und sei kurz und bündig und frage nie, wie du {name} helfen kannst oder was du für {name} tun kannst:\nCharlie: {style_example}\n\nVerstehst du das?',
            },
            {
                "role": "assistant",
                "content": f'Ja, ich verstehe. Bitte stelle mir den Dialogauszug und das gewünschte <msg style="..." length="..."></msg>-Tag zur Verfügung, damit ich es ausfüllen kann.',
            },
        ]
    else:
        assert False

    for message_pair in reversed(relevant_message_pairs):
        prompt += [
            {
                "role": "user",
                "content": f'{name}: {message_pair[language].msg_user}\nCharlie: <msg style="{message_pair[language].reply_style}" length="{message_pair[language].reply_length}"></msg>',
            },
            {
                "role": "assistant",
                "content": f"Charlie: {message_pair[language].msg_charlie_style}",
            },
        ]
    prompt.append(
        {
            "role": "user",
            "content": f'{name}: {text}\nCharlie: <msg style="{mood_style}" length="{message_length}"></msg>',
        }
    )

    print("DEBUG", prompt, mood_style, style_example)
    return prompt, mood_style, message_length


def extract_prompt_answers(full_answer):
    print("DEBUG", full_answer)
    answer_dict = {}
    # answer_dict["raw"] = re.search(
    #     "(?<=Charlie raw:)[\w\W\s]*(?=Charlie clean)", full_answer
    # )[0].strip()
    # answer_dict["clean"] = re.search(
    #     "(?<=Charlie clean:)[\w\W\s]*(?=Charlie style)", full_answer
    # )[0].strip()
    answer_dict["raw"] = None
    answer_dict["clean"] = None
    extracted_answer = re.search("(?<=Charlie:)[\w\W\s]*", full_answer)
    answer_dict["style"] = (
        extracted_answer[0] if extracted_answer is not None else full_answer
    ).strip()
    return answer_dict
