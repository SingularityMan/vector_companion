import os
import requests
import json
import re
import random
from collections import Counter
import time
from bs4 import BeautifulSoup
from datetime import datetime
import subprocess
import threading
import queue
import base64
from typing import Any, List, Tuple, Optional, Union, AsyncGenerator
import logging
import asyncio
from io import BytesIO

import numpy as np
import ollama
import aiohttp
import whisper
import pyaudio
import wave
import audioop
import simpleaudio as sa
import noisereduce as nr
from PIL import Image
import pyautogui as pygi
from transformers import AutoProcessor, AutoModelForCausalLM
import soundfile as sf
import torch

config_dir = os.path.dirname(os.path.realpath(__file__))
audio_transcriptions = ""

#------------------------------------------TEXT PROCESSING-----------------------------------------------------------#

class Agent():

    """
    Class of an agent that will be speaking. Contain their own name, gender, personality traits,
    system_prompts and dialogue_list.

    The class is responsible for generating a text response and summarizing the chat history when appropriate.
    """

    def __init__(self, agent_name, agent_gender, personality_traits, system_prompt1, system_prompt2, dialogue_list, language_model, speaker_wav, extraversion):
        self.agent_name = agent_name
        self.agent_gender = agent_gender
        self.system_prompt1 = system_prompt1
        self.system_prompt2 = system_prompt2
        self.previous_agent_message = ""
        self.personality_traits = personality_traits
        self.trait_set = []
        self.dialogue_list = dialogue_list
        self.language_model = language_model
        self.speaker_wav = speaker_wav
        self.extraversion = extraversion

    async def generate_text_stream(
    self,
    messages: list,
    agent_messages: list,
    system_prompt: str,
    user_input: str,
    context_length: int = 32000,
    temperature: float = 0.7,
    top_p: float = 0.3,
    top_k: int = 10000
    ) -> Tuple[List, List, AsyncGenerator[str, None]]:
        """
        Generates a text response as a stream using ollama.chat.
        Returns an asynchronous generator yielding sentences.
        """

        messages[0] = {"role": "system", "content": system_prompt}
        messages.append({"role": "user", "content": user_input})

        async def fetch_stream():
            loop = asyncio.get_event_loop()
            buffer = ''
            complete_response = ''  # To hold the entire concatenated output

            def run_chat():
                return ollama.chat(
                    model=self.language_model,
                    messages=messages,
                    stream=True,
                    keep_alive=-1,
                    options={
                        "repeat_penalty": 1.15,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "num_ctx": context_length,
                        "num_batch": 512,
                        "num_predict": 5000
                    }
                )

            # Run ollama.chat in the executor to prevent blocking the event loop
            stream = await loop.run_in_executor(None, run_chat)

            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    buffer += content
                    sentences, buffer = split_buffer_into_sentences(buffer)
                    for sentence in sentences:
                        # Clean up the sentence
                        sentence = clean_text(sentence)
                        complete_response += sentence + ' '  # Concatenate to the complete response
                        yield sentence

            # Handle any remaining buffer
            if buffer.strip():
                buffer = clean_text(buffer)
                complete_response += buffer + ' '  # Add remaining buffer to the complete response

            # Append the complete response as a single message
            complete_response = complete_response.strip()
            messages.append({"role": "assistant", "content": complete_response})
            agent_messages.append(
                f"Agent Name:{self.agent_name}, ({self.agent_gender})\nAgent Response: {complete_response}"
            )

        return messages, agent_messages, fetch_stream()
        
    def generate_text(
        self, messages: list,
        agent_messages: list,
        system_prompt: str,
        user_input: str,
        context_length: int = 32000,
        temperature: float = 0.7,
        top_p: float = 0.3,
        top_k: int = 10000
        ) -> Tuple[List, ...]:

        """
        Generates a text response.
        Additional response parameters may be added in the 'options' parameter in the payload variable.
        If len messages > 100: the entire chat history will be summarized.

        The response will be cleaned up to remove any unwanted characters.

        Parameters:

        - messages: Contains a list of the entire chat history.

        - agent_messages: contains a list of all the agent messages.

        - system_prompt: Text that guides the agent's response. Most LLMs store the instructions in user_input anyway.

        - user_input: The user's message. Also used to send instructions to the agent.

        - context_length: int containing length of the token context.

        - temperature: float containing the temperature. Accepts values between 0 and 1. Higher values make the response more random.

        - top_p: float value that selects the top n percent of probable tokens to choose from.

        - top_k: int value that selects the top n number (not percent!) of tokens most likely to be selected by the LLM's response.

        Returns a Tuple containing the messages, agent_messages in list format respectively and the text response.
        """

        url = "http://localhost:11434/api/chat"
        headers = {
            "Content-Type": "application/json"
        }

        messages[0] = {"role": "system", "content": system_prompt}
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.language_model,
            "messages": messages,
            "stream": False,
            "options":{
                "repeat_penalty": 1.15,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": context_length,
                "seed": random.randint(0, 2147483647),
                "num_batch": 512
                }
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            text_response = response_data.get("message", {}).get("content", "No response received")
            text_response = re.sub(r'"', '', text_response)
            text_response = re.sub(r'[^\x00-\x7F]+', '', text_response)
            text_response = re.sub(r'\(.*?\)', '', text_response)
            text_response = re.sub(r'\*.*?\*', '', text_response)
            text_response = text_response.replace('\\n', '')

            messages.append({"role": "assistant", "content": text_response})
            agent_messages.append(f"Agent Name:{self.agent_name}, ({self.agent_gender})\nAgent Response: {text_response}")
            return messages, agent_messages, text_response
        else:
            print("Failed to get a response. Status code:", response.status_code)
            print("Response:", response.text)
            return messages, agent_messages, "Failed to get a response."
        
class VectorAgent():

    """
    Class of a special agent that will help guide the response of the agents.
    The agent will be in charge of summarizing the immediate context and synthesizing contextual information.
    """

    def __init__(self, language_model):
        self.language_model = language_model
        pass

    def gather_agent_traits(self, agent_traits: list) -> str:

        """
        Gather the agent traits.
        Returns a string in the form of a concatenated list.
        """
        
        return ' '.join(agent_traits)

    def generate_text(
        self,
        messages: list,
        screenshot_description: str,
        audio_transcript_output: str,
        context_length: int
        ) -> str:

        """
        Generates a text response synthesizing information from various sources and contextualizing it.

        screenshot_description - Description of all the screenshots/OCR from screenshot_description.txt.
        audio_transcript_output - Audio transcript of PC audio output generated by whisper.
        context_length - int containing length of token processing.

        returns text response str
        """

        url = "http://localhost:11434/api/chat"
        headers = {
            "Content-Type": "application/json"
        }
            
        messages.append({"role": "user", "content":
                               "Review this contextual information:"
                               "\n\nScreenshot\OCR description: "+screenshot_description+
                               "\n\nAudio Transcript Output: "+audio_transcript_output+

                               "\n\nYour task will be the following: "

                               "\n\nGenerate a systematic list of meaningful key events based on the contextual information provided."
                               "\nYou must connect the key events together to interpret the overall situation in a big picture fashion."
                               "\nEach entry must explain how they relate to the overall situation."
                               "\nYou will sort the events in order of priority, from highest to lowest."
                               "\nWhat is the subject of the context?"
                               "\nWho are the most important entities or individuals in the situation?"
                               "\nWhat are the most significant parts of the context that must be taken into account?"
                               "\nEssentially you're explaining what is going on."
                               "\n\nComplete this task without mentioning it in any way. No acknowledgement, no offer of assistance, nothing. Just do it."})

        payload = {
            "model": self.language_model,
            "messages": messages,
            "stream": False,
            "options":{
                "temperature": 0.3,
                "num_ctx": context_length,
                "num_batch": 512
                }
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            text_response = response_data.get("message", {}).get("content", "No response received")
            text_response = re.sub(r'"', '', text_response)
            text_response = re.sub(r"\'", "", text_response)
            text_response = re.sub(r'[^\x00-\x7F]+', '', text_response)
            text_response = re.sub(r'\(.*?\)', '', text_response)
            text_response = re.sub(r'\*.*?\*', '', text_response)
            text_response = text_response.replace('\\n', '')
            print("[VECTOR INSTRUCTIONS]:", text_response)
            return text_response
        else:
            print("Failed to get a response. Status code:", response.status_code)
            print("Response:", response.text)
            return "Failed to get a response."

def split_buffer_into_sentences(buffer):
    """
    Splits buffer into sentences and returns the remaining buffer.
    """
    sentence_endings = re.compile(r'([.!?>])')
    sentences = []
    while True:
        match = sentence_endings.search(buffer)
        if match:
            end = match.end()
            sentence = buffer[:end].strip()
            sentences.append(sentence)
            buffer = buffer[end:]
        else:
            break
    return sentences, buffer

def clean_text(text):
    """
    Cleans the text by removing unwanted characters and patterns.
    """
    text = re.sub(r'"', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\*.*?\*', '', text)
    text = text.replace('\\n', '')
    return text.strip()

async def synthesize_sentence(tts, sentence, speaker_wav, sample_rate):
    """
    Asynchronously synthesizes a sentence and returns the audio data.
    """
    try:
        # Preprocess the sentence to remove problematic characters
        sentence = sentence.strip().strip("'\"")
        sentence = re.sub(r'[\'\"]', '', sentence)

        loop = asyncio.get_event_loop()
        # Synthesize the sentence using GPU
        audio = await loop.run_in_executor(
            None, lambda: tts.tts(text=sentence, speaker_wav=speaker_wav, language="en")
        )
        # Check the type of 'audio' and process accordingly
        if isinstance(audio, list):
            # Convert list to NumPy array
            audio_array = np.array(audio, dtype=np.float32)
        elif isinstance(audio, np.ndarray):
            # If it's already a NumPy array
            audio_array = audio.astype(np.float32)
        else:
            raise TypeError(f"Unexpected audio type: {type(audio)}")

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
        y=audio_array,
        sr=sample_rate,
        prop_decrease=0.0,
        freq_mask_smooth_hz=1000,
        time_mask_smooth_ms=100,
        stationary=True,
        n_fft=512,
        win_length=512,
        use_torch=True,
        device='cuda'
    )

        return reduced_noise
        #return audio
    except Exception as e:
        print(f"Error during TTS synthesis for sentence '{sentence}': {e}")
        return None

def find_repeated_words(text: str, threshold: int = 6) -> str:
    pattern = r'\b(\w+)\b'
    words = re.findall(pattern, text, re.IGNORECASE)
    word_counts = Counter(words)
    filtered_words = [word for word in words if word_counts[word.lower()] < threshold]
    word_counts.clear()
    return " ".join(filtered_words)

def remove_repetitive_phrases(text: str) -> str:
    pattern = re.compile(r'(\b\w+\b(?:\s+\b\w+\b){0,4})(?:\s+\1)+', re.IGNORECASE)
    result = pattern.sub(r'\1', text)
    return result

def check_sentence_length(text: str, sentence_length: int = 2) -> Union[Tuple[List[str], str], str]:
    text = remove_repetitive_phrases(text)
    sentences = re.split(r'(?:(?<=[.?;!]))\s', text.strip())
    if sentences:
        if len(sentences) > 1:
            return (sentences[:sentence_length], ' '.join(sentences[:sentence_length]))
        else:
            return sentences, sentences[0]
    return "No valid text found."

#----------------------------------------IMAGE PROCESSING----------------------------------------------#

image_lock = False

# Vision model view images:
def view_image(vision_model: Any, processor: Any):

    global image_lock
        
    if not image_lock:

        try:
            image_lock = True

            image_picture = pygi.screenshot("axiom_screenshot.png")
            with open("axiom_screenshot.png", "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            prompt = "Provide as concise a summary as possible of what you see on the screen." 

            # Generate the response
            result = ollama.generate(
                model="minicpm-v:8b-2.6-q4_0",
                prompt=prompt,
                keep_alive=-1,
                images=[encoded_image],
                options={
                    "repeat_penalty": 1.15,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 512,
                    "num_predict": 500
                    }
                )
            
            current_time = datetime.now().time()

            text_response = result["response"]
            with open("screenshot_description.txt", "a", encoding='utf-8') as f:
                    f.write(f"\n\nScreenshot Contents at {current_time.strftime('%H:%M:%S')}: \n\n"+text_response)


            time.sleep(10)
            
            image_lock = False
            
        except Exception as e:
            image_lock = False
            print("Error:", e)
            pass

#-------------------------------------------------AUDIO PROCESSING---------------------------------------------#

def record_audio(
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
    processor: str,
    can_speak_event: bool
    ) -> Optional[bool]:

    global image_lock
    
    ii = 0
    recording_index = 0
    
    try:
        while True:

            # Cancel recording if Agent speaking
            if not can_speak_event.is_set():
                time.sleep(0.05)
                print("[record_user_mic] Waiting for response to complete...")
                continue
            
            # Start Recording
            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                input_device_index=3,
                                frames_per_buffer=CHUNK
                                )
            frames = []
            image_path = None

            # Record for RECORD_SECONDS
            silence_start = None
            recording_started = False
            if not image_lock:
                print("[SCREENSHOT TAKEN]", ii)
                threading.Thread(target=view_image, args=(vision_model, processor)).start()

            while True:
                
                if not can_speak_event.is_set():
                    time.sleep(0.05)
                    if not can_speak_event.is_set():
                        print("Cancelling recording, agent is speaking.")
                        break
                        #stream.stop_stream()
                        #stream.close()
                        
                        #return False

                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except IOError as e:
                    print(f"Error reading audio stream: {e}")
                    continue

                rms = audioop.rms(data, 2)  # width=2 for format=paInt16

                #print("[NOT SPEAKING]:", rms)
                if ii < int(RATE / CHUNK * RECORD_SECONDS):
                    ii += 1
                    if ii % (int(RATE / CHUNK)/15) == 0:
                        if not image_lock:
                            print("[SCREENSHOT TAKEN]", ii)
                            threading.Thread(target=view_image, args=(vision_model, processor)).start()

                if rms >= THRESHOLD: #(ii >= int(RATE / CHUNK * RECORD_SECONDS)):
                    silence_start = time.time()
                    if not recording_started:
                        SILENCE_LIMIT = 0.75
                        recording_start_time = time.time()
                        print("recording...")
                        if not image_lock:
                            print("[SCREENSHOT TAKEN]", ii)
                            threading.Thread(target=view_image, args=(vision_model, processor)).start()
                        THRESHOLD = 85
                        recording_started = True
                    elif rms >= THRESHOLD and recording_started:
                        #print(f"[CONTINUING TO SPEAK]:", rms)
                        silence_start = time.time()
                        can_speak_event.set()
                        
                        if time.time() - recording_start_time >= 30:

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
                                input_device_index=3,
                                frames_per_buffer=CHUNK
                                )
                            recording_index += 1
                            recording_start_time = time.time()
                            frames = []
                            
                if rms < THRESHOLD and recording_started:
                    if time.time() - silence_start > SILENCE_LIMIT:
                        print("finished recording")
                        can_speak_event.clear()
                        break
                    
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

def record_audio_output(
                        audio: str,
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

    while True:

        # Check if an agent is responding.
        if not can_speak_event.is_set():
            print("[record_audio_output] Waiting for response to complete...")
            time.sleep(1)
            continue

        # Create a PyAudio instance
        p = pyaudio.PyAudio()

        # Find the device index of the VB-Cable adapter
        device_index = None
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            #print(device_info)
            if 'CABLE Output (VB-Audio Virtual' in device_info['name']:  # Look for 'VB-Audio' instead of 'VB-Cable'
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

            #print("FRAMES DIALOGUE OUTPUT:"+str(i)+"/"+str(int(RATE / CHUNK * RECORD_SECONDS)))

            if not can_speak_event.is_set():
                time.sleep(0.05)
                break
                    
            data = stream.read(CHUNK, exception_on_overflow=True)
            frames.append(data)

        print("* done recording Audio Transcript")
        file_index += 1

        # Stop the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the audio to a .wav file
        wf = wave.open('audio_transcript_output{}.wav'.format(file_index), 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        for file in os.listdir(os.getcwd()):
            if "audio_transcript_output" in file:
                file_path = os.path.join(os.getcwd(), file)
                if os.path.isfile(file_path):
                    audio_transcript_output = transcribe_audio(model, model_name, file_path, probability_threshold=0.5)
                    if len(audio_transcript_output.strip().split()) <= 6:
                        audio_transcript_output = ""
                    audio_transcriptions += " "+audio_transcript_output
                    audio_transcriptions = audio_transcriptions.strip()
                else:
                    print("No audio transcribed")
                    #audio_transcriptions = ""

        if file_index >= file_index_count:
            can_speak_event.clear()
            
        if not can_speak_event.is_set():
            break

        frames = []

def transcribe_audio(model: Any, model_name, WAVE_OUTPUT_FILENAME: str, RATE: int = 16000, probability_threshold=0.2) -> str:

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
    #print(f"Detected language: {detected_language}")

    if 'en' not in detected_language and ('turbo' not in model_name and 'large' not in model_name):
        return ""

    options = whisper.DecodingOptions(
    task="translate",
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
    if "audio_transcript_output" in WAVE_OUTPUT_FILENAME:
        if len(user_voice_output.strip().split()) < 6:
            user_voice_output = ""
    try:
        os.remove(WAVE_OUTPUT_FILENAME)
    except Exception as e:
        print("Error:", e)
        user_voice_output = ""
        return user_voice_output
    
    # Print the recognized text
    #print(user_voice_output)
    return user_voice_output
