from __future__ import annotations
import asyncio

import os
import wave
import audioop
import pyaudio
import whisper

from config.image_processing import image_lock, run_image_description
from config.text_processing import remove_repetitive_phrases, remove_repetitive_phrases, find_repeated_words

audio_transcriptions = ""

async def record_audio(
    audio: str,
    WAVE_OUTPUT_FILENAME: str,
    FORMAT: int,
    RATE: int,
    CHANNELS: int,
    CHUNK: int,
    RECORD_SECONDS: int,
    THRESHOLD: int,
    SILENCE_LIMIT: int,
    vision_model: str,
    vision_context_length,
    can_speak_event: bool
    ) -> Optional[bool]:

    global image_lock

    image_lock = False
    
    recording_index = 0
    loop = asyncio.get_event_loop()
    
    try:
        while True:

            # Create a PyAudio instance
            p = pyaudio.PyAudio()

            # Find the device index of the VB-Cable adapter
            device_index = None
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                #print(device_info)
                if 'Microphone' in device_info['name'] and 'Microsoft' not in device_info['name']:  # Look for 'VB-Audio' instead of 'VB-Cable'
                    print("[FOUND MICROPHONE. DEVICE INDEX SET TO]", i)
                    device_index = i
                    break

            # Cancel recording if Agent speaking
            if not can_speak_event.is_set():
                await asyncio.sleep(0.05)
                print("[record_user_mic] Waiting for response to complete...")
                continue
            
            # Start Recording
            stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                input_device_index=device_index,
                                frames_per_buffer=CHUNK
                                )
            frames = []
            image_path = None

            # Record for RECORD_SECONDS
            silence_start = None
            recording_started = False

            while True:

                # Spawn image description task only once for this session.
                if not image_lock:
                    image_lock = True
                    image_desc_task = asyncio.create_task(
                        run_image_description(vision_model, vision_context_length, can_speak_event)
                    )
                
                if not can_speak_event.is_set():
                    await asyncio.sleep(0.05)
                    if not can_speak_event.is_set():
                        # Cancel image description task and break out
                        if not image_desc_task.done():
                            image_desc_task.cancel()
                            try:
                                await image_desc_task
                            except asyncio.CancelledError:
                                pass
                        image_lock = False
                        print("Cancelling recording, agent is speaking.")
                        break
                        
                                
                try:
                    data = await loop.run_in_executor(None, stream.read, CHUNK, True)
                except IOError as e:
                    print(f"Error reading audio stream: {e}")
                    continue

                rms = audioop.rms(data, 2)  # width=2 for format=paInt16

                if rms >= THRESHOLD: #(ii >= int(RATE / CHUNK * RECORD_SECONDS)):
                    silence_start = loop.time()
                    if not recording_started:
                        SILENCE_LIMIT = 0.75
                        recording_start_time = loop.time()
                        print("recording...")
                        THRESHOLD = 65
                        recording_started = True
                    elif rms >= THRESHOLD and recording_started:
                        silence_start = loop.time()
                        can_speak_event.set()
                        
                        if loop.time() - recording_start_time >= 30:

                            # Stop Recording
                            stream.stop_stream()
                            stream.close()

                            # Write your new .wav file with built-in Python 3 Wave module
                            waveFile = wave.open(WAVE_OUTPUT_FILENAME+str(recording_index)+".wav", 'wb')
                            waveFile.setnchannels(CHANNELS)
                            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                            waveFile.setframerate(RATE)
                            waveFile.writeframes(b''.join(frames))
                            waveFile.close()

                            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                input_device_index=device_index,
                                frames_per_buffer=CHUNK
                                )
                            recording_index += 1
                            recording_start_time = loop.time()
                            frames = []
                            
                if rms < THRESHOLD and recording_started:
                    if loop.time() - silence_start > SILENCE_LIMIT:
                        print("finished recording")
                        can_speak_event.clear()
                        break
                    
                if recording_started:    
                    frames.append(data)

            if not can_speak_event.is_set():

                # Stop Recording
                stream.stop_stream()
                stream.close()

                # Write your new .wav file with built-in Python 3 Wave module
                waveFile = wave.open(WAVE_OUTPUT_FILENAME+str(recording_index)+".wav", 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()

                recording_index += 1
                
                return True

    except Exception as e:
        print(f"An error occurred in record_audio: {e}")
        return None

async def record_audio_output(
                        audio: str,
                        listen_for_audio: bool,
                        WAVE_OUTPUT_FILENAME: str,
                        FORMAT: int, CHANNELS: int,
                        RATE: int,
                        CHUNK: int,
                        RECORD_SECONDS: int,
                        file_index_count: int,
                        can_speak_event: bool,
                        model: Any,
                        model_name: str
                        ):

    file_index = 0
    global audio_transcriptions
    audio_transcriptions = ""
    loop = asyncio.get_event_loop()

    while True:

        # Create a PyAudio instance
        p = pyaudio.PyAudio()

        # Check if an agent is responding.
        if not can_speak_event.is_set():
            print("[record_audio_output] Waiting for response to complete...")
            await asyncio.sleep(1)
            continue

        # Find the device index of the VB-Cable adapter
        device_index = None
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            #print(device_info)
            if 'CABLE Output' in device_info['name']:  # Look for 'VB-Audio' instead of 'VB-Cable'
                device_index = i
                break

        if device_index is None:
            print("Could not find VB-Cable device")
            exit(1)

        # Open the stream for recording
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=device_index)

        print("* recording Audio Transcript")

        frames = []

        # Record the audio
        FRAMES_PER_SECOND = int(RATE / CHUNK)*2
        print("Frames per second:", FRAMES_PER_SECOND)

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

            if not can_speak_event.is_set():
                await asyncio.sleep(0.01)
                break
                    
            try:
                data = await loop.run_in_executor(None, stream.read, CHUNK, True)
            except Exception as e:
                print(f"Error reading stream: {e}")
                continue
            frames.append(data)

        print("* done recording Audio Transcript")
        file_index += 1

        # Stop the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the audio to a .wav file in a non-blocking way
        def save_wav():
            wf = wave.open(f'audio_transcript_output{file_index}.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
        await loop.run_in_executor(None, save_wav)

        for file in os.listdir(os.getcwd()):
            if "audio_transcript_output" in file:
                file_path = os.path.join(os.getcwd(), file)
                if os.path.isfile(file_path):
                    audio_transcript_output = await loop.run_in_executor(
                        None, lambda: transcribe_audio(model, model_name, file_path, RATE=16000, probability_threshold=0.7)
                    )
                    if len(audio_transcript_output.strip().split()) <= 3:
                        audio_transcript_output = ""
                    audio_transcriptions += " "+audio_transcript_output
                    audio_transcriptions = audio_transcriptions.strip()

        print("File index", file_index)
        print("Audio Transcript Output:", audio_transcript_output)

        if file_index >= file_index_count and audio_transcript_output.strip() != "" and not listen_for_audio:
            can_speak_event.clear()
        elif audio_transcript_output.strip() == "" and listen_for_audio: 
            can_speak_event.clear()

        await asyncio.sleep(0.05)
            
        if not can_speak_event.is_set():
            break

        frames = []

def transcribe_audio(model: Any, model_name, WAVE_OUTPUT_FILENAME: str, RATE: int = 16000, probability_threshold=0.5) -> str:

    """
    Transcribes audio via whisper

    Parameters:

    - model: whisper model

    - WAVE_OUTPUT_FILENAME: the file name of the audio output.

    - RATE: rate to determine max_audio_length

    returns a string containing the user audio output.
    """

    audio_data = whisper.load_audio(WAVE_OUTPUT_FILENAME)
    audio_data = whisper.pad_or_trim(audio_data)

    max_audio_length = 30 * RATE
    audio_data = audio_data[:max_audio_length]
    
    n_mels = 128 if 'turbo' in model_name or 'large' in model_name else 80
    mel = whisper.log_mel_spectrogram(audio_data, n_mels=n_mels).to(model.device)

    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    if 'en' not in detected_language and ('turbo' not in model_name and 'large' not in model_name):
        return ""

    options = whisper.DecodingOptions(
    task="transcribe",
    language="en",
    prompt=None,
    prefix=None,
    suppress_blank=False,
    fp16=True,
    )

    result = whisper.decode(model, mel, options)
    print("[NO SPEECH PROBABILITY]:", result.no_speech_prob)
    if result.no_speech_prob > probability_threshold:
        user_voice_output = ""
        return user_voice_output
        
    user_voice_output_raw = result.text
    user_voice_output = find_repeated_words(user_voice_output_raw)
    user_voice_output = remove_repetitive_phrases(user_voice_output)
    
    try:
        os.remove(WAVE_OUTPUT_FILENAME)
    except Exception as e:
        print("Error:", e)
        user_voice_output = ""
        return user_voice_output
    
    # Print the recognized text
    return user_voice_output
