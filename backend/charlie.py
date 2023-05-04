import os
import collections
import re
import json
import hashlib
import datetime

import utils.helper_functions as uhf
import utils.data_structs as uds

import numpy as np
import time
import deepl
import openai


class Charlie:
    def __init__(
        self,
        session_token="",
        no_init=False,
        base_config=None,
        socketio=None,
        persistent_memory_session=False,
        memory_database=None,
    ):
        print("DEBUG init charlie with session token", session_token)
        self.session_token = session_token
        self.memory_database = memory_database
        self.initialized = False
        if no_init:
            return
        # load config file
        self.config = json.load(
            open(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
                "r",
            )
        )
        self.api_keys = json.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "api_keys.json"
                ),
                "r",
            )
        )
        if base_config is not None:
            self.config["base"] = base_config

        self.socketio = socketio
        self.logger = uhf.Logger(
            self.session_token,
            self.config["base"]["userUID"],
            socketio,
            persistent_memory_session=persistent_memory_session,
        )

        self.name = self.config["base"]["name"]
        self.mode = uds.Mode.DORMANT
        self.gender = uds.Gender.get(self.config["base"]["gender"])
        self.gender_user = uds.Gender.get(self.config["base"]["gender-user"])
        self.language = uds.Language.get(self.config["base"]["language"])
        self.translation_language_source = uds.Language.ENGLISH
        self.translation_language_target = uds.Language.GERMAN
        self.mood = uhf.Mood(
            style_en=self.config["base"]["style_en"],
            situation_en=self.config["base"]["situation_en"],
            logger=self.logger,
        )
        self.tts_method = self.config["base"]["tts-method"]

        self._init_openai()
        self._init_tts()
        self.translation_model = self._init_deepl()

        self.audio_processor = uhf.AudioProcessor(logger=self.logger)
        self.memory_buffer = collections.deque([], maxlen=200)
        self.memory_buffer_remember_count = self.config["base"]["memory_size"]

        self.remote_controlled = True

        self.initialized = True

    def _init_tts(self):
        self.logger.log(uds.Mode.SYSTEM, "system", f"Initialize Google TTS API..")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.api_keys["api_keys"]["google"],
        )
        os.environ["ELEVENLABS_TTS_CREDENTIALS"] = self.api_keys["api_keys"][
            "elevenlabs-tts"
        ]

    def _init_deepl(self):
        self.logger.log(uds.Mode.SYSTEM, "system", "Initialize DeepL API..")
        return deepl.Translator(self.api_keys["api_keys"]["deepl"])

    def _init_openai(self):
        self.logger.log(uds.Mode.SYSTEM, "system", "Initialize OpenAI API..")
        openai.api_key = self.api_keys["api_keys"]["openai"]

    def _init_audio_stream(self, state):
        import pyaudio

        self.logger.log(
            uds.Mode.SYSTEM,
            "system",
            "Initialize audio stream (error warnings can be ignored as long as it doesn't crash)..",
        )
        recording_params = uds.RecordingParams(self.config)
        # Calculate buffer and roll lengths in samples
        buffer_length_samples = int(
            recording_params.buffer_length_seconds * recording_params.sample_rate
        )

        # Initialize circular buffer with zeros
        buffer = collections.deque(maxlen=buffer_length_samples)
        buffer.extend(np.zeros(buffer_length_samples))

        p = pyaudio.PyAudio()

        # Define callback function to read audio data into circular buffer
        def callback(in_data, frame_count, time_info, status):
            nonlocal state

            if state.is_recording:
                state.recorded_frames += frame_count
            data = np.frombuffer(in_data, dtype=np.int16)
            buffer.extend(data)
            return (in_data, pyaudio.paContinue)

        # Open audio stream and start recording
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=recording_params.sample_rate,
            input=True,
            frames_per_buffer=recording_params.chunk_size,
            stream_callback=callback,
        )

        return p, stream, buffer, recording_params

    def _process_audio_input(self):
        # transform the deque to a numpy array and normalize
        buffer_arr = uhf.buffer_to_numpy(
            self.audio_buffer, self.recording_state.recorded_frames
        )
        # transcribe the audio buffer
        input_text = uhf.transcribe_audio_buffer_api(
            buffer_arr, self.recording_params.sample_rate
        )
        self.logger.track_stats(
            "whisper", "", buffer_arr.shape[0] / self.recording_params.sample_rate
        )

        handle_settings_change_result = self._accept_text_input(input_text)
        if handle_settings_change_result is not None:
            return handle_settings_change_result

        return self._process_text_input()

    def _process_recorded_audio(self):
        input_text = uhf.transcribe_audio_buffer_api(
            self.audio_processor.recording_buffer, self.audio_processor.sample_rate
        )
        self.logger.track_stats(
            "whisper",
            "",
            self.audio_processor.recording_buffer.shape[0]
            / self.audio_processor.sample_rate,
        )
        # self.audio_processor.save_recording_to_file(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         "logfiles",
        #         f"{input_text[:10].replace(' ', '').replace('.','').replace(',', '').replace('!','')}.wav",
        #     )
        # )
        if input_text is None or input_text == "":
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Empty or null input text retrieved from audio",
            )
            return None

        handle_settings_change_result = self._accept_text_input(input_text)
        if handle_settings_change_result is not None:
            return handle_settings_change_result

        return self.logger.socketio.start_background_task(self._process_text_input)

    def _handle_settings_change(
        self, settings_changed, settings_output, settings_list, input_text
    ):
        if self.current_custom_username is None:
            self.current_custom_username = self.name
        # either settings_changed is None or settings_changed is true
        if settings_changed is not None:
            self.logger.log(uds.Mode.SYSTEM, self.current_custom_username, input_text)
            if settings_list is not None:
                self.logger.log(uds.Mode.SYSTEM, "system", f"{settings_list}")
        else:
            self.logger.log(uds.Mode.SYSTEM, self.current_custom_username, input_text)
            return True

        output_text = settings_output
        self.logger.log(uds.Mode.SYSTEM, "Charlie", output_text)
        if self.tts_method != "notts":
            with uhf.suppress_stdout():
                uhf.text_to_speech_api(
                    self.tts_method,
                    self.language
                    if self.mode != uds.Mode.TRANSLATION
                    else self.translation_language_target,
                    self.gender,
                    output_text,
                    None,
                    self.session_token,
                    self.logger,
                )
                self.logger.track_stats(self.tts_method, output_text)

        return False

    def _accept_text_input(self, input_text, custom_username: str | None = None):
        settings_changed, settings_output, settings_list = self._catch_systems_input(
            input_text
        )
        if settings_changed is None or settings_changed:
            print(
                "DEBUG: systems input caught",
                settings_changed,
                settings_output,
                settings_list,
            )
            self.current_input_text = None
            self.current_custom_username = self.name
            return self._handle_settings_change(
                settings_changed, settings_output, settings_list, input_text
            )

        self.current_custom_username = (
            custom_username if custom_username is not None else self.name
        )
        print(f"DEBUG log with custom username {self.current_custom_username}")
        self.logger.log(self.mode, self.current_custom_username, input_text)
        self.current_input_text = input_text

        return None

    def _process_text_input(self):
        if self.current_input_text is None:
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Attempted to process current input text that is None..",
            )
            return False

        if self.mode == uds.Mode.TRANSLATION:
            output_text = uhf.translate_transcript(
                self.translation_model,
                self.current_input_text,
                self.translation_language_source,
                self.translation_language_target,
            ).text
            self.logger.track_stats("deepl", self.current_input_text)
            self.logger.log(self.mode, "Charlie", output_text)
        elif self.mode == uds.Mode.CONVERSATION:
            output_text_dict, reply_style, reply_length = uhf.prompt_gpt(
                self.mode,
                self.current_input_text,
                self.language,
                self.current_custom_username,
                self.translation_model,
                self.memory_buffer,
                self.memory_buffer_remember_count,
                self.mood,
                self.logger,
                self.memory_database,
            )
            # we can't track here cause we already extracted most of the info inside the prompt_gpt function
            output_text = self._post_process_text_output(output_text_dict)
            self.logger.log(self.mode, "Charlie", output_text)
            self.memory_buffer.append(
                {
                    self.language: uds.MessagePair(
                        self.current_input_text,
                        output_text_dict,
                        reply_style,
                        reply_length,
                    )
                }
            )
        elif self.mode == uds.Mode.INFORMATION:
            output_text, reply_style = uhf.prompt_gpt(
                self.mode,
                self.current_input_text,
                self.language,
                self.current_custom_username,
                self.translation_model,
                self.memory_buffer,
                self.config["base"]["memory_size"],
                self.logger,
            )
            # we can't track here cause we already extracted most of the info inside the prompt_gpt function
            self.logger.log(self.mode, "Charlie", output_text)
        else:
            # voice error message
            output_text = uhf.get_error_message(self.translation_model, self.language)
            self.logger.log(self.mode, "system", output_text)

        if output_text == "":
            return False

        if self.tts_method != "notts":
            with uhf.suppress_stdout():
                if self.remote_controlled:
                    self.logger.socketio.start_background_task(
                        uhf.text_to_speech_api,
                        self.tts_method,
                        self.language
                        if self.mode != uds.Mode.TRANSLATION
                        else self.translation_language_target,
                        self.gender,
                        output_text.replace(",", ""),
                        self.logger.socketio,
                        self.session_token,
                        self.logger,
                    )
                else:
                    uhf.text_to_speech_api(
                        self.tts_method,
                        self.language
                        if self.mode != uds.Mode.TRANSLATION
                        else self.translation_language_target,
                        self.gender,
                        output_text.replace(",", ""),
                        None,
                        self.session_token,
                        None,
                    )
            self.logger.track_stats(self.tts_method, output_text.replace(",", ""))

        return False

    def _post_process_text_output(self, output_text_dict):
        # Post processing of conversation text output to make it sound more natural
        print("DEBUG: Output string raw:", output_text_dict)
        if output_text_dict["style"] is not None:
            output_text = output_text_dict["style"]
        elif output_text_dict["none"] is not None:
            output_text = output_text_dict["none"]
        else:
            return "Something went wrong when post processing text output"
        processed_output_text = output_text.replace("\n", " ")
        processed_output_text = re.sub(
            f"(?<=[\w])[\W]*\s{self.current_custom_username}(?=\W)",
            "",
            processed_output_text,
        )
        processed_output_text = re.sub(
            f"^{self.current_custom_username}\W\s+", "", processed_output_text
        )
        processed_output_text = re.sub(
            f'^{self.current_custom_username}:\s+"', "", processed_output_text
        )
        # processed_output_text = re.sub(f'".*$', "", processed_output_text)
        print("DEBUG: Processed outout:", processed_output_text)
        return processed_output_text

    def _catch_systems_input(self, input_text):
        settings_change_requested = False
        if uhf.simple_match_shutdown(input_text):
            return None, "", None
        elif uhf.simple_match_list_settings(input_text):
            settings_results_string = f"The current mode is {self.mode}. The gender is {self.gender}. The base language is {self.language}. The translation source language is {self.translation_language_source}. The translation target language is {self.translation_language_target}."
            return True, settings_results_string, None
        elif uhf.simple_match_change_settings(input_text):
            settings_change_requested = True
            settings_input = uhf.clean_settings_input(input_text)

        if not settings_change_requested:
            return False, "", None
        settings_list = uhf.prompt_gpt_settings(settings_input, self.language)
        settings_results_string = self._process_settings_list(settings_list)
        return True, settings_results_string, settings_list

    def _process_settings_list(self, settings_list):
        settings_results_string = ""
        # array [new mode, new gender, new source language, new target language]
        # non requested items are None, rest is Mode, string or Language

        # first change mode and languages before getting the strings
        if settings_list[0] is not None:
            self.mode = settings_list[0]

        update_tts_language = False
        if settings_list[2] is not None:
            if settings_list[0] == uds.Mode.TRANSLATION:
                if self.translation_language_source != settings_list[2]:
                    update_tts_language = True
                    self.translation_language_source = settings_list[2]
            else:
                if self.language != settings_list[2]:
                    update_tts_language = True
                    self.language = settings_list[2]

        if settings_list[3] is not None:
            if self.translation_language_target != settings_list[3]:
                update_tts_language = True
                self.translation_language_target = settings_list[3]

        if settings_list[0] is not None:
            if settings_list[0] == uds.Mode.TRANSLATION:
                output_language = self.translation_language_target
            elif settings_list[0] == uds.Mode.CONVERSATION:
                output_language = self.language
            elif settings_list[0] == uds.Mode.INFORMATION:
                output_language = self.language
            else:
                assert False
            settings_results_string += (
                uhf.get_settings_result_string(
                    self.translation_model, "mode", settings_list[0], output_language
                )
                + ". "
            )
        else:
            output_language = (
                self.language
                if self.mode != uds.Mode.TRANSLATION
                else self.translation_language_target
            )

        if settings_list[1] is not None:
            self.gender = settings_list[1]
            output_language = (
                self.language
                if self.mode != uds.Mode.TRANSLATION
                else self.translation_language_target
            )
            settings_results_string += (
                uhf.get_settings_result_string(
                    self.translation_model, "gender", settings_list[1], output_language
                )
                + ". "
            )

        if settings_list[2] is not None:
            if settings_list[0] == uds.Mode.TRANSLATION:
                output_language = (
                    self.language
                    if self.mode != uds.Mode.TRANSLATION
                    else self.translation_language_target
                )
                settings_results_string += (
                    uhf.get_settings_result_string(
                        self.translation_model,
                        "language_source",
                        settings_list[2],
                        output_language,
                    )
                    + ". "
                )
            else:
                output_language = (
                    self.language
                    if self.mode != uds.Mode.TRANSLATION
                    else self.translation_language_target
                )
                settings_results_string += (
                    uhf.get_settings_result_string(
                        self.translation_model,
                        "language",
                        settings_list[2],
                        output_language,
                    )
                    + ". "
                )

                # go through memory buffer and translate the messages if necessary
                # (after a language change)
                for i in range(
                    min(self.config["base"]["memory_size"], len(self.memory_buffer))
                ):
                    print(f"DEBUG: check msg {i} for translation need")
                    if self.language not in self.memory_buffer[-i]:
                        print(f"DEBUG: translate msg {i}")
                        source_language = list(self.memory_buffer[-i].keys())[0]

                        # translate message
                        translated_message_pair = uds.MessagePair()
                        if self.memory_buffer[-i][source_language].msg_user is not None:
                            translated_message_pair.msg_user = uhf.translate_transcript(
                                self.translation_model,
                                self.memory_buffer[-i][source_language].msg_user,
                                source_language,
                                self.language,
                            ).text
                            self.logger.track_stats(
                                "deepl",
                                self.memory_buffer[-i][source_language].msg_user,
                            )
                        else:
                            translated_message_pair.msg_user = None

                        if (
                            self.memory_buffer[-i][source_language].msg_charlie
                            is not None
                        ):
                            translated_message_pair.msg_charlie = (
                                uhf.translate_transcript(
                                    self.translation_model,
                                    self.memory_buffer[-i][source_language].msg_charlie,
                                    source_language,
                                    self.language,
                                ).text
                            )
                            self.logger.track_stats(
                                "deepl",
                                self.memory_buffer[-i][source_language].msg_charlie,
                            )
                        else:
                            translated_message_pair.msg_charlie = None

                        if (
                            self.memory_buffer[-i][source_language].msg_charlie_raw
                            is not None
                        ):
                            translated_message_pair.msg_charlie_raw = (
                                uhf.translate_transcript(
                                    self.translation_model,
                                    self.memory_buffer[-i][
                                        source_language
                                    ].msg_charlie_raw,
                                    source_language,
                                    self.language,
                                ).text
                            )
                            self.logger.track_stats(
                                "deepl",
                                self.memory_buffer[-i][source_language].msg_charlie_raw,
                            )
                        else:
                            translated_message_pair.msg_charlie_raw = None

                        if (
                            self.memory_buffer[-i][source_language].msg_charlie_clean
                            is not None
                        ):
                            translated_message_pair.msg_charlie_clean = (
                                uhf.translate_transcript(
                                    self.translation_model,
                                    self.memory_buffer[-i][
                                        source_language
                                    ].msg_charlie_clean,
                                    source_language,
                                    self.language,
                                ).text
                            )
                            self.logger.track_stats(
                                "deepl",
                                self.memory_buffer[-i][
                                    source_language
                                ].msg_charlie_clean,
                            )
                        else:
                            translated_message_pair.msg_charlie_clean = None

                        if (
                            self.memory_buffer[-i][source_language].msg_charlie_style
                            is not None
                        ):
                            translated_message_pair.msg_charlie_style = (
                                uhf.translate_transcript(
                                    self.translation_model,
                                    self.memory_buffer[-i][
                                        source_language
                                    ].msg_charlie_style,
                                    source_language,
                                    self.language,
                                ).text
                            )
                            self.logger.track_stats(
                                "deepl",
                                self.memory_buffer[-i][
                                    source_language
                                ].msg_charlie_style,
                            )
                        else:
                            translated_message_pair.msg_charlie_style = None

                        if (
                            self.memory_buffer[-i][source_language].reply_style
                            is not None
                        ):
                            translated_message_pair.reply_style = (
                                uhf.translate_transcript(
                                    self.translation_model,
                                    self.memory_buffer[-i][source_language].reply_style,
                                    source_language,
                                    self.language,
                                ).text
                            )
                            self.logger.track_stats(
                                "deepl",
                                self.memory_buffer[-i][source_language].reply_style,
                            )
                        else:
                            translated_message_pair.reply_style = None

                        if (
                            self.memory_buffer[-i][source_language].reply_length
                            is not None
                        ):
                            translated_message_pair.reply_length = (
                                uhf.translate_transcript(
                                    self.translation_model,
                                    self.memory_buffer[-i][
                                        source_language
                                    ].reply_length,
                                    source_language,
                                    self.language,
                                ).text
                            )
                            self.logger.track_stats(
                                "deepl",
                                self.memory_buffer[-i][source_language].reply_length,
                            )
                        else:
                            translated_message_pair.reply_length = None

                        self.memory_buffer[-i][self.language] = translated_message_pair

        if settings_list[3] is not None:
            output_language = (
                self.language
                if self.mode != uds.Mode.TRANSLATION
                else self.translation_language_target
            )
            settings_results_string += (
                uhf.get_settings_result_string(
                    self.translation_model,
                    "language_target",
                    settings_list[3],
                    output_language,
                )
                + ". "
            )

        return settings_results_string

    def start_local_conversation(self):
        self.remote_controlled = False
        self.mode = uds.Mode.CONVERSATION

        self.recording_state = uhf.RecordingState()
        (
            self.pyaudio_obj,
            self.audio_stream,
            self.audio_buffer,
            self.recording_params,
        ) = self._init_audio_stream(self.recording_state)

        self.logger.log(uds.Mode.SYSTEM, "system", "Ready to record..")
        self.audio_stream.start_stream()

        stop_conversation = False
        while True:
            # the conditional checks are horrendous but who cares about performance
            time.sleep(0.1)
            buffer_arr = np.asarray(self.audio_buffer)
            if (
                not self.recording_state.is_recording
                and np.abs(
                    buffer_arr[-self.recording_params.activation_window :]
                ).mean()
                > self.recording_params.activation_threshold
            ):
                self.logger.log(uds.Mode.SYSTEM, "system", "Start recording..")
                self.recording_state.is_recording = True
                self.recording_state.recorded_frames = (
                    self.recording_params.activation_window
                )
            elif (
                self.recording_state.is_recording
                and self.recording_state.recorded_frames
                > self.recording_params.deactivation_window
                and np.abs(
                    buffer_arr[-self.recording_params.deactivation_window :]
                ).mean()
                <= self.recording_params.deactivation_threshold
            ):
                self.recording_state.is_recording = False
                self.logger.log(
                    uds.Mode.SYSTEM,
                    "system",
                    f"Deactivate recording, recorded frames: {self.recording_state.recorded_frames} = {self.recording_state.recorded_frames / self.recording_params.sample_rate} seconds",
                )
                stop_conversation = self._process_audio_input()
                if not stop_conversation:
                    self.logger.log(uds.Mode.SYSTEM, "system", "Ready to record..")
            if stop_conversation:
                break
        self.end_local_conversation()

    def end_local_conversation(self):
        self.logger.log(uds.Mode.SYSTEM, "system", "Shut down Charlie..")
        self.mode = uds.Mode.DORMANT
        # Stop recording and close audio stream
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pyaudio_obj.terminate()
        self.remote_controlled = True

    ####################################################################
    #####     API      #################################################
    ####################################################################

    def initialize_conversation(
        self,
        session_token="",
        base_config=None,
        socketio=None,
        persistent_memory_session=False,
    ):
        self.__init__(
            session_token=session_token,
            base_config=base_config,
            socketio=socketio,
            persistent_memory_session=persistent_memory_session,
            memory_database=self.memory_database,
        )
        self.mode = uds.Mode.CONVERSATION
        if base_config["gender"] == "male":
            socketio.emit("live2dchangemodelmale", self.session_token)
        else:
            socketio.emit("live2dchangemodelfemale", self.session_token)
        self.logger.log(uds.Mode.SYSTEM, "system", "Ready to accept input..")

    def end_conversation(self):
        if not self.initialized:
            # TODO: Give this info back to the client
            return
        if self.mode == uds.Mode.DORMANT:
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Attempted to end conversation while already dormant..",
            )
            return
        self.logger.log(uds.Mode.SYSTEM, "system", "Shut down Charlie..")
        self.mode = uds.Mode.DORMANT

    def get_last_message(self):
        if not self.initialized:
            # TODO: Give this info back to the client
            return
        return self.logger.get_last_log_message()

    def append_and_process_audio_buffer_arr(self, buffer_arr):
        event = self.audio_processor.append_and_process(buffer_arr)
        # event = "START", "STOP", "STARTSTOP", None
        return event

    def process_external_audio_input(self):
        if not self.initialized:
            # TODO: Give this info back to the client
            return
        if self.mode == uds.Mode.DORMANT:
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Attempted to process external audio while dormant..",
            )
            return

        self._process_recorded_audio()

    def accept_external_text_input(self, text, custom_username: str | None = None):
        if not self.initialized:
            # TODO: Give this info back to the client
            print("Charlie is not yet initiliazed when accepting external text input")
            return
        if self.mode == uds.Mode.DORMANT:
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Attempted to accept external text while dormant..",
            )
            return
        return self._accept_text_input(text, custom_username)

    def process_external_text_input(self):
        if not self.initialized:
            # TODO: Give this info back to the client
            return
        if self.mode == uds.Mode.DORMANT:
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Attempted to process external text while dormant..",
            )
            return

        self._process_text_input()

    def set_settings_manually(self, settings_list):
        if not self.initialized:
            # TODO: Give this info back to the client
            return
        if self.mode == uds.Mode.DORMANT:
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Attempted to set settings manually while dormant..",
            )
            return
        # accepts an array/list of the form
        # ["CONVERSATION" | "INFORMATION" | "TRANSLATION" | None, "male" | "female" | None, Language=string | None, Language=string | None]

        settings_output = self._process_settings_list(settings_list)

        return self._handle_settings_change(
            True,
            settings_output,
            settings_list,
            "Manually changed settings",
        )

    def update_config(self, new_config):
        if not self.initialized:
            # TODO: Give this info back to the client
            return
        if self.mode == uds.Mode.DORMANT:
            self.logger.log(
                uds.Mode.SYSTEM,
                "system",
                "Attempted to set settings manually while dormant..",
            )
            return

        self.config["base"] = new_config
        self.name = self.config["base"]["name"]
        self.gender = uds.Gender.get(self.config["base"]["gender"])
        self.gender_user = uds.Gender.get(self.config["base"]["gender-user"])
        self.language = uds.Language.get(self.config["base"]["language"])
        self.mood = uhf.Mood(
            style_en=self.config["base"]["style_en"],
            situation_en=self.config["base"]["situation_en"],
            logger=self.logger,
        )
        self.memory_buffer_remember_count = self.config["base"]["memory_size"]
        self.tts_method = self.config["base"]["tts-method"]

        if self.gender == "male":
            self.socketio.emit("live2dchangemodelmale", self.session_token)
        else:
            self.socketio.emit("live2dchangemodelfemale", self.session_token)

        self.logger.log(
            uds.Mode.SYSTEM, "system", f"Changed config to {self.config['base']}"
        )


class CharlieSession:
    def __init__(self, user_uid: str, persistent: bool = False):
        self.user_uid: str = user_uid
        self.session_token: str = self._create_session_token(user_uid)
        self.last_update: datetime.datetime = datetime.datetime.today()
        memory_database = None
        if persistent:
            memory_vector_db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logfiles",
                user_uid,
                "persistent_session",
                "memory_vector_db.pickle",
            )
            memory_summary_db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logfiles",
                user_uid,
                "persistent_session",
                "memory_summary_db.pickle",
            )
            memory_database = uhf.MemoryDatabase(
                memory_vector_db_path, memory_summary_db_path
            )
            print(
                f"DEBUG initialized memory database with {memory_database.memory_vector_db.shape[0]} memories"
            )
        self.charlie_instance: Charlie = Charlie(
            session_token=self.session_token,
            no_init=True,
            persistent_memory_session=persistent,
            memory_database=memory_database,
        )
        self.charlie_is_responsive: bool = True
        self.persistent_memory_session: bool = persistent

    def _create_session_token(self, user_uid):
        timestamp_str = str(datetime.datetime.today().timestamp())
        h = hashlib.new("sha256")
        h.update((user_uid + timestamp_str).encode())
        return h.hexdigest()


if __name__ == "__main__":
    charlie = Charlie()
    charlie.start_local_conversation()
    # charlie.initialize_conversation()
    # charlie.process_external_text_input("Hey Charlie, how are you?")
    # charlie.set_settings_manually(["INFORMATION", "male", "DE", None])
