import os
import sys
import models
import pyaudio
import numpy as np
import simpleaudio as sa
import collections
import time
from contextlib import contextmanager


def main():
    init_cli()
    start_listening()


def play_test_audio():
    # calculate note frequencies
    A_freq = 440
    Csh_freq = A_freq * 2 ** (4 / 12)
    E_freq = A_freq * 2 ** (7 / 12)

    # get timesteps for each sample, T is note duration in seconds
    sample_rate = 44100
    T = 0.25
    t = np.linspace(0, T, int(T * sample_rate), False)

    # generate sine wave notes
    A_note = np.sin(A_freq * t * 2 * np.pi)
    Csh_note = np.sin(Csh_freq * t * 2 * np.pi)
    E_note = np.sin(E_freq * t * 2 * np.pi)

    # concatenate notes
    audio = np.hstack((A_note, Csh_note, E_note))
    # normalize to 16-bit range
    audio *= 32767 / np.max(np.abs(audio))
    # convert to 16-bit data
    audio = audio.astype(np.int16)

    # start playback
    print(audio.shape)
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

    # wait for playback to finish before exiting
    play_obj.wait_done()


def init_cli():
    print("Prepare models...")
    models.prepare_models()


def start_listening(
    buffer_length_seconds=30,
    sample_rate=16000,
    chunk_size=1024,
    activation_threshold=30,
    activation_window=8000,
    deactivation_window=32000,
):
    # Calculate buffer and roll lengths in samples
    buffer_length_samples = int(buffer_length_seconds * sample_rate)
    is_recording = False
    recorded_frames = 0

    # Initialize circular buffer with zeros
    buffer = collections.deque(maxlen=buffer_length_samples)
    buffer.extend(np.zeros(buffer_length_samples))

    # Initialize PyAudio object
    p = pyaudio.PyAudio()

    # Define callback function to read audio data into circular buffer
    def callback(in_data, frame_count, time_info, status):
        nonlocal is_recording
        nonlocal recorded_frames
        nonlocal buffer

        if is_recording:
            recorded_frames += frame_count
        data = np.frombuffer(in_data, dtype=np.int16)
        buffer.extend(data)
        return (in_data, pyaudio.paContinue)

    # Open audio stream and start recording
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
        stream_callback=callback,
    )

    stream.start_stream()
    print("Ready to record...")

    # Continuously read data from the circular buffer and yield a rolling window
    while True:
        # Yield a rolling window of the circular buffer
        # data = stream.read(chunk_size)
        time.sleep(0.1)
        buffer_arr = np.asarray(buffer)
        if (
            not is_recording
            and np.abs(buffer_arr[-activation_window:]).mean() > activation_threshold
        ):
            print("Activate recording")
            is_recording = True
            recorded_frames = activation_window
        elif (
            is_recording
            and recorded_frames > deactivation_window
            and np.abs(buffer_arr[-deactivation_window:]).mean() <= activation_threshold
        ):
            is_recording = False
            print(
                f"Deactivate recording, recorded frames: {recorded_frames} = {recorded_frames / sample_rate} seconds"
            )
            dispatch_to_assistant(buffer, recorded_frames)

    # Stop recording and close audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()


def dispatch_to_assistant(buffer, recorded_frames):
    # transform the deque to a numpy array and normalize
    buffer_arr = buffer_to_numpy(buffer, recorded_frames)
    # transcribe the audio buffer
    result = models.transcribe_audio_buffer(buffer_arr)
    print("Input:", result.text)
    if False:
        # translate the input to english
        transcript_translated = models.translate_transcript(result.text)
        print("Output:", transcript_translated.text)
        # text to speech english
        with suppress_stdout():
            sample_rate, voice_arr = models.text_to_speech(transcript_translated.text)
    else:
        # prompt gpt to get an answer
        gpt_answer = models.prompt_gpt(result.text)
        print("Output:", gpt_answer)
        # text to speech english
        with suppress_stdout():
            sample_rate, voice_arr = models.text_to_speech(gpt_answer)

    audio = voice_arr * 32767.0 / np.max(np.abs(voice_arr))
    # convert to 16-bit data
    audio = audio.astype(np.int16)

    # start playback
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)


def buffer_to_numpy(buffer, recorded_frames):
    # np.frombuffer(buffer, np.int16).flatten().astype(np.float32) / 32768.0
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


if __name__ == "__main__":
    main()
