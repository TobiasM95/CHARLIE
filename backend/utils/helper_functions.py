from contextlib import contextmanager
import pickle
import sys
import os
import io
import requests
import re
from datetime import datetime, timedelta
import json
import wave

import numpy as np
from scipy.spatial import distance
import openai
import google.cloud.texttospeech as tts
import soundfile as sf
from thefuzz import fuzz

from .data_structs import Language, Mode, Gender, gpt_models

from .conversation_prompts.iterative_refinement_v2 import (
    get_conversation_prompt_chat_gpt,
    extract_prompt_answers,
    chat_gpt_parameter_dict,
)

openai.api_key = json.load(
    open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api_keys.json"),
        "r",
    )
)["api_keys"]["openai"]

####################################################################
#####  COMPLEX CLASSES  ############################################
####################################################################


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

    def __init__(
        self,
        style_en="a witty sarcastic friend",
        situation_en="hanging out together",
        logger=None,
    ):
        self.mood: Mood = Mood.NEUTRAL
        self.style: dict[Language, str] = {}
        self.style[Language.ENGLISH] = style_en
        self.style_example: dict[Language, str] = {}
        self.style_example[Language.ENGLISH] = self._get_style_example(style_en, logger)
        self.situation: dict[Language, str] = {}
        self.situation[Language.ENGLISH] = situation_en
        self.logger: Logger = logger
        logger.log(
            Mode.SYSTEM,
            "system",
            f"Style example: {self.style_example[Language.ENGLISH]}",
        )

    def _get_style_example(self, style_en, logger):
        style_prompt = [
            {
                "role": "system",
                "content": "You are an actor and renowned expert in roleplaying different styles that are given to you.",
            },
            {
                "role": "user",
                "content": f'How would you write a stereotypical response with the style "{style_en}" to the sentence "Hey, wanna hang out?" in text? If you cannot due to your guidelines then give a response that is still within your guidelines instead, i.e. tone down the style a little bit. Answer with exactly one sentence in quotation marks.',
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

    def get_style(self, translation_model, language):
        if language not in self.style or language not in self.style_example:
            self._translate_style(translation_model, language)
        return self.style[language], self.style_example[language]

    def get_situation(self, translation_model, language):
        if language not in self.situation:
            self._translate_situation(translation_model, language)
        return self.situation[language]

    def _translate_style(self, translation_model, language):
        if language not in self.style:
            self.style[language] = translate_transcript(
                translation_model,
                self.style[Language.ENGLISH],
                Language.ENGLISH,
                language,
            ).text
            self.logger.track_stats("deepl", self.style[Language.ENGLISH])
        if language not in self.style_example:
            self.style_example[language] = translate_transcript(
                translation_model,
                self.style_example[Language.ENGLISH],
                Language.ENGLISH,
                language,
            ).text
            self.logger.track_stats("deepl", self.style_example[Language.ENGLISH])

    def _translate_situation(self, translation_model, language):
        if language not in self.situation:
            self.situation[language] = translate_transcript(
                translation_model,
                self.situation[Language.ENGLISH],
                Language.ENGLISH,
                language,
            ).text
            self.logger.track_stats("deepl", self.situation[Language.ENGLISH])

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


class MemoryDatabase:
    def __init__(
        self,
        memory_vector_db_path: str,
        memory_summary_db_path: str,
        learning_phase: int = 5,
    ):
        self.memory_vector_db_path = memory_vector_db_path
        self.memory_summary_db_path = memory_summary_db_path
        if os.path.isfile(memory_vector_db_path):
            self.memory_vector_db = pickle.load(open(self.memory_vector_db_path, "rb"))
        else:
            self.memory_vector_db = np.zeros((0, 1536))
        if os.path.isfile(memory_summary_db_path):
            self.memory_summary_db = pickle.load(
                open(self.memory_summary_db_path, "rb")
            )
        else:
            self.memory_summary_db = []

        # How many memories have to exist before retrieving returns some of them
        self.learning_phase: int = learning_phase

    def insert_memory(self, embedding: list[float], summary: str) -> bool:
        if any(np.equal(self.memory_vector_db, embedding).all(1)):
            return False
        self.memory_vector_db = np.vstack(
            (self.memory_vector_db, np.asarray(embedding).reshape((1, -1)))
        )
        self.memory_summary_db.append(summary)
        return True

    def retrieve_memory(self, embedding: list[float], num_memories: int):
        if self.memory_vector_db.shape[0] < self.learning_phase:
            return []
        if num_memories > self.memory_vector_db.shape[0]:
            num_memories = self.memory_vector_db.shape[0]
        if num_memories == 1:
            return [
                self.memory_summary_db[
                    distance.cdist(
                        self.memory_vector_db,
                        np.asarray(embedding).reshape((1, -1)),
                        "cosine",
                    ).argmin()
                ]
            ]
        queried_indices = np.argpartition(
            distance.cdist(
                self.memory_vector_db, np.asarray(embedding).reshape((1, -1)), "cosine"
            ).flatten(),
            list(range(num_memories)),
        )[:num_memories]
        queried_memories = []
        for qi in queried_indices:
            queried_memories.append(self.memory_summary_db[qi])
        return queried_memories

    def save_to_disk(self):
        pickle.dump(self.memory_vector_db, open(self.memory_vector_db_path, "wb"))
        pickle.dump(self.memory_summary_db, open(self.memory_summary_db_path, "wb"))


####################################################################
#####     LOGGING      #############################################
####################################################################


class Logger:
    def __init__(
        self, session_token, userUID, socketio=None, persistent_memory_session=False
    ):
        self.split_log_message_threshold = 10
        self.split_log_timedelta = timedelta(seconds=60)
        self.last_log_split_timestamp = datetime.now()

        self.userUID = userUID
        self.session_token = session_token
        self.persistent_memory_session = persistent_memory_session
        self.stats_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "stats"
        )
        os.makedirs(self.stats_dir, exist_ok=True)
        self.npm_logfile_dir = None
        self.pm_logfile_dir = None
        self.filename = None
        if userUID is not None:
            self.npm_logfile_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "logfiles",
                userUID,
                "npm_sessions",
            )
            self.pm_logfile_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "logfiles",
                userUID,
                "persistent_session",
                "raw_logfiles",
            )
            os.makedirs(self.npm_logfile_dir, exist_ok=True)
            os.makedirs(self.pm_logfile_dir, exist_ok=True)
            self.session_time = datetime.now()
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            weekday = datetime.now().strftime("%A")

            self.filename = os.path.join(
                self.pm_logfile_dir
                if persistent_memory_session
                else self.npm_logfile_dir,
                f"session_{timestamp}.txt".replace(":", "-"),
            )
            self.last_log_message = f"[{timestamp}][SYSTEM, system] Start session on {weekday} at {timestamp}, session token {self.session_token}, persistency={self.persistent_memory_session}."
            print(self.last_log_message)
            with open(self.filename, "w", encoding="utf-8") as logfile:
                logfile.write(self.last_log_message + "\n")

        self.socketio = socketio
        if socketio is not None and session_token is not None:
            print("emit conversationmessage", self.session_token, self.last_log_message)
            self.socketio.emit(
                "convmsg",
                {"session_token": self.session_token, "message": self.last_log_message},
            )

        self.stats = self._load_stats()

    def _load_stats(self):
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

    def debug_log(self, log_message, suffix=""):
        uid: str = self.userUID if self.userUID is not None else ""
        if suffix != "":
            suffix = "_" + suffix
        os.makedirs(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "logfiles",
                uid,
            ),
            exist_ok=True,
        )
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "logfiles",
                uid,
                f"debug_log{suffix}.txt",
            ),
            "+a",
            encoding="utf-8",
        ) as debug_logfile:
            debug_logfile.write(log_message + "\n")
        print(f"DEBUG LOG: {log_message}")

    def log(self, mode, name, text, verbose=True):
        if (
            self.persistent_memory_session
            and (datetime.now() - self.last_log_split_timestamp)
            > self.split_log_timedelta
        ):
            self.last_log_split_timestamp = datetime.now()
            self.split_log()
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.last_log_message = f"[{timestamp}][{mode}, {name}] {text}"
        if verbose:
            print(self.last_log_message)
        with open(self.filename, "a", encoding="utf-8") as logfile:
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
        with open(self.filename, "r", encoding="utf-8") as logfile:
            for line in logfile:
                print(line)

    def split_log(self):
        if not os.path.isfile(self.filename):
            return
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        weekday = datetime.now().strftime("%A")
        dir_path, filename_w_ext = os.path.split(os.path.abspath(self.filename))
        filename_wo_ext = filename_w_ext[:-4]
        with open(self.filename, "r", encoding="utf-8") as logfile:
            lines = logfile.readlines()
        if (
            len(lines) == 0
            or len([l for l in lines if "SYSTEM, " not in l])
            < self.split_log_message_threshold
        ):
            return
        first_line = lines[0]

        session_start_timestamp_match = re.match("\[(.*?)\]", first_line)
        session_token_match = re.match("session token (.*?),", first_line)
        if session_token_match is not None:
            session_token = session_token_match.group(1)
        else:
            session_token = "INVALIDATEDSESSIONTOKEN"
        if session_start_timestamp_match is None:
            first_line = f"[{timestamp}][SYSTEM, system] Start session on {weekday} at {timestamp}, session token {session_token}, persistency={self.persistent_memory_session}.\n"
            session_start = timestamp
        else:
            session_start = session_start_timestamp_match.group(1)
            first_line = f"[{session_start}][SYSTEM, system] Start session on {weekday} at {timestamp}, session token {session_token}, persistency={self.persistent_memory_session}.\n"
        session_start_formatted = session_start.replace(":", "-")
        split_count = len(
            [
                n
                for n in next(os.walk(dir_path))[2]
                if session_start_formatted + "_split" in n
            ]
        )
        os.rename(
            os.path.abspath(self.filename),
            os.path.join(dir_path, filename_wo_ext + f"_split_{split_count}.txt"),
        )

        with open(self.filename, "w", encoding="utf-8") as newfile:
            newfile.write(first_line)
            newfile.write(f"[{timestamp}][SYSTEM, system] Continued session logfile.\n")

        print(
            f"DEBUG split log at {os.path.join(dir_path, filename_wo_ext + f'_split_{split_count}.txt')}"
        )
        self.debug_log(
            f"DEBUG split log at {os.path.join(dir_path, filename_wo_ext + f'_split_{split_count}.txt')}",
            "memorization",
        )

    def track_stats(self, api, message="", duration=0.0, tokens=0):
        self.stats = self._load_stats()
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
    mode: Mode,
    text: str,
    language: Language,
    name: str,
    translation_model,
    memory_buffer=None,
    remembered_message_count: int = 3,
    mood: Mood | None = None,
    logger: Logger | None = None,
    memory_database: MemoryDatabase | None = None,
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
            logger,
            {
                "situation-description": mood.get_situation(
                    translation_model, language
                ),
                "memory-database": memory_database,
            },
        )
        try:
            result = openai.ChatCompletion.create(
                model=gpt_models[mode],
                messages=chat_gpt_messages,
                max_tokens=250,
                **chat_gpt_parameter_dict,
            )
        except openai.error.RateLimitError as e:
            err_msg = str(e)
            if "exceeded your current quota" in err_msg:
                ret_msg = "I think our time together came to an end. I have to say goodbye now."
                return (
                    {
                        "none": ret_msg,
                        "raw": ret_msg,
                        "clean": ret_msg,
                        "style": ret_msg,
                    },
                    "rate quota exception answer style",
                    "rate quota exception reply length",
                )
            else:
                ret_msg = "Wow you're talking so fast, let me think about that for a moment..."
                return (
                    {
                        "none": ret_msg,
                        "raw": ret_msg,
                        "clean": ret_msg,
                        "style": ret_msg,
                    },
                    "rate exception answer style",
                    "rate exception reply length",
                )
        except Exception as e:
            err_msg = str(e)
            ret_msg = (
                "I think something went terribly wrong here. We need an adult in here."
            )
            return (
                {"none": ret_msg, "raw": ret_msg, "clean": ret_msg, "style": ret_msg},
                "general exception answer style",
                "general exception reply length",
            )
        logger.track_stats(api="chatgpt", tokens=result["usage"]["total_tokens"])
        logger.debug_log(result["choices"][0]["message"]["content"])
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
        model=gpt_models[Mode.INFORMATION],
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


def prompt_gpt_summarization(
    conversation_string: str, weekday: str, date: str, logger: Logger
):
    prompt = [
        {
            "role": "user",
            "content": f'This conversation happened on {weekday}, {date}. Summarize the content like it was memory database entry of Charlie (gender neutral). Keep it short. Include important events, information and key words and make sure for long conversations that you include all information. At the end, add tags that include activities, events, key words and emotions. Output format should be "Content:" and "Tags:" on a single line each.\n\n"'
            + conversation_string
            + '"',
        }
    ]
    try:
        result = openai.ChatCompletion.create(
            model=gpt_models[Mode.INFORMATION],
            messages=prompt,
            max_tokens=1000,
            temperature=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
    except Exception as e:
        err_msg = str(e)
        return False, err_msg
    logger.track_stats(api="chatgpt", tokens=result["usage"]["total_tokens"])
    return True, result["choices"][0]["message"]["content"]


def get_text_embedding(summary):
    # TODO: Track usage for embeddings
    return openai.Embedding.create(input=summary, model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


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


def memorize_conversations(active_logfiles: list[str], logger: Logger):
    logger.debug_log(
        f"THREAD DEBUG attempt memorization while active convos exist: {active_logfiles}",
        "memorization",
    )
    SUMMARIZATION_CHUNK_MAX_CHARACTER_COUNT = 12000
    sessions_to_memorize = {}

    # search which non-active sessions have to be memorized
    logfile_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "logfiles"
    )
    os.makedirs(logfile_dir, exist_ok=True)
    user_dirs = next(os.walk(logfile_dir))[1]
    for user_dir in user_dirs:
        logger.debug_log(
            f"THREAD DEBUG check {user_dir} for logfiles..", "memorization"
        )
        raw_logfiles_dir = os.path.join(
            logfile_dir, user_dir, "persistent_session", "raw_logfiles"
        )
        os.makedirs(raw_logfiles_dir, exist_ok=True)
        leftover_persistent_sessions = next(os.walk(raw_logfiles_dir))[2]

        for leftover_session in leftover_persistent_sessions:
            logger.debug_log(
                f"THREAD DEBUG check if {leftover_session} can be memorized..",
                "memorization",
            )
            file_path = os.path.join(raw_logfiles_dir, leftover_session)
            if file_path not in active_logfiles:
                logger.debug_log(
                    f"THREAD DEBUG path {file_path} IS READY +++", "memorization"
                )
                if user_dir not in sessions_to_memorize:
                    sessions_to_memorize[user_dir] = [leftover_session]
                else:
                    sessions_to_memorize[user_dir] += [leftover_session]
            else:
                logger.debug_log(
                    f"THREAD DEBUG path {file_path} IS NOT READY ---", "memorization"
                )

    # for each session to memorize
    for uuid, session_filenames in sessions_to_memorize.items():
        memory_vector_db_path = os.path.join(
            logfile_dir, uuid, "persistent_session", "memory_vector_db.pickle"
        )
        memory_summary_db_path = os.path.join(
            logfile_dir, uuid, "persistent_session", "memory_summary_db.pickle"
        )
        memory_database = MemoryDatabase(memory_vector_db_path, memory_summary_db_path)
        for session_filename in session_filenames:
            session_file_path = os.path.join(
                logfile_dir,
                uuid,
                "persistent_session",
                "raw_logfiles",
                session_filename,
            )
            with open(
                session_file_path,
                "r",
            ) as session_file:
                # gather meta information like date, day of week, etc.
                meta_info = session_file.readline()
                weekday_match = re.search(
                    "(?:Start session on )(.*?)(?:\s.*$)", meta_info
                )
                date_match = re.search(
                    "(?:Start session on )(?:.*?)(?:at )(.*?)(?:T.*$)", meta_info
                )
                if weekday_match is None or date_match is None:
                    continue
                weekday = weekday_match.group(1)
                date = date_match.group(1)
                # filter out system messages
                # split the conversations in chunks that suit chatgpt summarization
                filtered_message_chunks = []
                chunk = ""
                chunk_length = 0
                for line in session_file:
                    if "[SYSTEM, " in line:
                        continue
                    name_match = re.search("(?:\[CONVERSATION, )(.*?)(?:\])", line)
                    content_match = re.search("(?:\[.*?\]\[.*?\] )(.*)", line)
                    if name_match is None or content_match is None:
                        continue
                    name = name_match.group(1)
                    content = content_match.group(1)
                    line_length = len(name) + len(content) + 3
                    chunk += name + ": " + content + "\n"
                    chunk_length += line_length
                    if chunk_length >= SUMMARIZATION_CHUNK_MAX_CHARACTER_COUNT:
                        filtered_message_chunks.append(chunk)
                        chunk = ""
                        chunk_length = 0
                        continue
                if chunk_length > 0:
                    filtered_message_chunks.append(chunk)

            session_file_success = True
            db_has_changed = False
            for chunk in filtered_message_chunks:
                # let ChatGPT summarize the content (user info and charlie info has to be included)
                success, summary = prompt_gpt_summarization(
                    chunk, weekday, date, logger
                )
                if not success:
                    session_file_success = False
                    break
                summary = "Date: " + weekday + ", " + date + ":\n" + summary
                # create openai vector embedding
                embedding = get_text_embedding(summary)
                db_has_changed = memory_database.insert_memory(embedding, summary)
            if not session_file_success:
                break
            if db_has_changed:
                memory_database.save_to_disk()
            os.remove(session_file_path)


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
