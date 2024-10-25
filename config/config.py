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
import base64
from typing import Any, List, Tuple, Optional

import whisper
import pyaudio
import wave
import audioop
import simpleaudio as sa
from PIL import Image
import pyautogui as pygi
from transformers import AutoProcessor, AutoModelForCausalLM
import soundfile as sf

config_dir = os.path.dirname(os.path.realpath(__file__))

can_speak_event = threading.Event()
can_speak_event.set()

#------------------------------------------TEXT PROCESSING-----------------------------------------------------------#

class Agent():

    """
    Class of an agent that will be speaking. Contain their own name, gender, personality traits,
    system_prompts and dialogue_list.

    The class is responsible for generating a text response and summarizing the chat history when appropriate.
    """

    def __init__(self, agent_name, agent_gender, personality_traits, system_prompt1, system_prompt2, dialogue_list):
        self.agent_name = agent_name
        self.agent_gender = agent_gender
        self.system_prompt1 = system_prompt1
        self.system_prompt2 = system_prompt2
        self.previous_agent_message = ""
        self.personality_traits = personality_traits
        self.trait_set = []
        self.dialogue_list = dialogue_list

    def summarize_conversation(self, agent_messages: list, context_length: int) -> Tuple[list, str]:

        """
        Summarizes the conversation if it reaches 100 messages.

        Parameters:

        - agent_messages: list of messages generated by the agents.
        
        - context_length: int containing length of the token context.

        returns a tuple containing the updated agent_messages and the text response of the summary.
        """
        
        url = "http://localhost:11434/api/chat"
        headers = {
            "Content-Type": "application/json"
        }
        
        summary_prompt = "Provide an expository summary of this conversation in list format, highlighting the most significant events that occurred while being as objective as possible:\n\n" + "\n".join(agent_messages[-32000:])

        summary_payload = {
            "model": "llama3.1:8b-instruct-q4_0",
            "messages": [{"role": "system", "content": "Summarize the conversation in list format."}, {"role": "user", "content": summary_prompt}],
            "stream": False,
            "options": {
                "repeat_penalty": 1.15,
                "num_ctx": context_length
            }
        }

        response = requests.post(url, headers=headers, data=json.dumps(summary_payload))

        if response.status_code == 200:
            response_data = response.json()
            text_response = response_data.get("message", {}).get("content", "No response received")
            agent_messages.append(text_response)
            return agent_messages, text_response
        else:
            print("Failed to get a summary response. Status code:", response.status_code)
            print("Response:", response.text)
            return agent_messages, "Failed to get a summary response."
        
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

        if len(messages) > 100:
            
            print("[MESSAGE LIMIT EXCEEDED. SUMMARRIZING CONVERSATION...]")
            
            messages = [{"role": "system", "content": system_prompt}]
            agent_messages, conversation_summary = self.summarize_conversation(agent_messages, 32000)
            messages.append({"role": "user", "content": conversation_summary})
            
            print("[CONVERSATION SUMMARY]:", conversation_summary)

        messages[0] = {"role": "system", "content": system_prompt}
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "llama3.1:8b-instruct-q4_0",
            "messages": messages,
            "stream": False,
            "options":{
                "repeat_penalty": 1.15,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": context_length,
                "seed": random.randint(0, 2147483647)
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

    def __init__(self):
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
            "model": "llama3.1:8b-instruct-q4_0",
            "messages": messages,
            "stream": False,
            "options":{
                "temperature": 0.3,
                "num_ctx": context_length
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
            print("[VECTOR INSTRUCTIONS]:", text_response)
            return text_response
        else:
            print("Failed to get a response. Status code:", response.status_code)
            print("Response:", response.text)
            return "Failed to get a response."


def find_repeated_words(text: str, threshold: int = 6) -> str:
    pattern = r'\b(\w+)\b'
    words = re.findall(pattern, text, re.IGNORECASE)
    word_counts = Counter(words)
    filtered_words = [word for word in words if word_counts[word.lower()] < threshold]
    word_counts.clear()
    return " ".join(filtered_words)

def remove_repetitive_phrases(text: str) -> Tuple[list, str]:
    pattern = re.compile(r'(\b\w+\b(?:\s+\b\w+\b){0,4})(?:\s+\1)+', re.IGNORECASE)
    result = pattern.sub(r'\1', text)
    return result

def check_sentence_length(text: str, sentence_length: int = 2) -> Tuple[list, str]:
    text = remove_repetitive_phrases(text)
    sentences = re.split(r'(?:(?<=[.?;!]))\s', text.strip())
    if sentences:
        if len(sentences) > 1:
            return (sentences[:sentence_length], ' '.join(sentences[:sentence_length]))
        else:
            return sentences, sentences[0]
    return "No valid text found."

def remove_repetitive_phrases(text: str, max_repeats: int = 3) -> str:
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        phrase = [words[i]]
        count = 1
        for j in range(1, len(words) - i):
            if words[i:i+j] == words[i+j:i+2*j]:
                phrase = words[i:i+j]
                count += 1
            else:
                break
        result.extend(phrase * min(count, max_repeats))
        i += len(phrase) * count
    return ' '.join(result)

#----------------------------------------IMAGE PROCESSING----------------------------------------------#

image_lock = False

# Vision model view images:
def view_image(vision_model: Any, processor: Any):

    """
    Views a screenshot and captions it to provide a description.
    """

    global image_lock

    try:

        image_lock = True

        prompt = "<MORE_DETAILED_CAPTION>"

        #image_picture = pygi.screenshot("axiom_screenshot.png")

        image_file = "axiom_screenshot.png"
        image = Image.open(image_file)

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = vision_model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1000,
            do_sample=True,
            num_beams=10
        )

        generated_ids.to('cpu')
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))

        # Get the current time
        current_time = datetime.now().time()

        with open("screenshot_description.txt", "a", encoding='utf-8') as f:
            f.write(f"\n\nScreenshot Contents at {current_time.strftime('%H:%M:%S')}: \n\n"+parsed_answer['<MORE_DETAILED_CAPTION>'])

        view_image_ocr(vision_model, processor)

        image_lock = False
    except Exception as e:
        image_lock = False
        print("Error:", e)
        pass

def view_image_ocr(vision_model: Any, processor: Any):

    """
    Views an image and attempts to extract OCR.
    """

    try:

        prompt = "<OCR_WITH_REGION>"

        image_file = "axiom_screenshot.png"
        image = Image.open(image_file)

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = vision_model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=500,
            do_sample=False,
            temperature=1,
            num_beams=5
        )

        generated_ids.to('cpu')
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task='<OCR_WITH_REGION>', 
            image_size=(image.width, image.height)
        )

        current_time = datetime.now().time()

        with open("screenshot_description.txt", "a", encoding='utf-8') as f:
                f.write(f"\n\nOCR text in image at {current_time.strftime('%H:%M:%S')}:\n\n"+" ".join(parsed_answer['<OCR_WITH_REGION>']["labels"]))
    except Exception as e:
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
    processor: str
    ) -> Optional[bool]:

    global image_lock
    
    ii = 0
    
    try:
        while True:

            # Cancel recording if Agent speaking
            if not can_speak_event.is_set():
                time.sleep(1)
                print("[record_user_mic] Waiting for response to complete...")
                continue
            
            # Start Recording
            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                input_device_index=1,
                                frames_per_buffer=CHUNK
                                )
            frames = []
            image_path = None

            # Record for RECORD_SECONDS
            silence_start = None
            recording_started = False

            while True:
                
                if not can_speak_event.is_set():
                    print("Cancelling recording, agent is speaking.")
                    stream.stop_stream()
                    stream.close()
                    time.sleep(0.25)
                    return False

                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except IOError as e:
                    print(f"Error reading audio stream: {e}")
                    continue

                rms = audioop.rms(data, 2)  # width=2 for format=paInt16
                if ii < int(RATE / CHUNK * RECORD_SECONDS):
                    ii += 1
                    if ii % (int(RATE / CHUNK)/15) == 0:
                        if not image_lock:
                            print("[SCREENSHOT TAKEN]", ii)
                            image_picture = pygi.screenshot("axiom_screenshot.png")
                            threading.Thread(target=view_image, args=(vision_model, processor)).start()

                if rms >= THRESHOLD or (ii >= int(RATE / CHUNK * RECORD_SECONDS)):
                    silence_start = time.time()
                    if not recording_started:
                        SILENCE_LIMIT = 1
                        print("recording...")
                        image_picture = pygi.screenshot("axiom_screenshot.png")
                        if not image_lock:
                            print("[SCREENSHOT TAKEN]", ii)
                            threading.Thread(target=view_image, args=(vision_model, processor)).start()
                        THRESHOLD = 150
                        recording_started = True
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
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()

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
                        file_index_count: int
                        ):

    file_index = 0

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
            if 'VB-Audio' in device_info['name']:  # Look for 'VB-Audio' instead of 'VB-Cable'
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
                        input_device_index=3)

        print("* recording Audio Transcript")

        frames = []

        # Record the audio
        FRAMES_PER_SECOND = int(RATE / CHUNK)*2
        print("Frames per second:", FRAMES_PER_SECOND)

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

            #print("FRAMES DIALOGUE OUTPUT:"+str(i)+"/"+str(int(RATE / CHUNK * RECORD_SECONDS)))

            if not can_speak_event.is_set():
                time.sleep(1)
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

        if file_index >= file_index_count:
            can_speak_event.clear()
            
        if not can_speak_event.is_set():
            break

        frames = []

def transcribe_audio(model: Any, WAVE_OUTPUT_FILENAME: str, RATE: int = 16000) -> str:

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

    mel = whisper.log_mel_spectrogram(audio_data).to(model.device)

    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    if 'en' not in detected_language:
        return ""

    options = whisper.DecodingOptions(
    task="transcribe",
    prompt=None,
    prefix=None,
    suppress_blank=False,
    fp16=True  
    )

    result = whisper.decode(model, mel, options)
    user_voice_output_raw = result.text
    user_voice_output = find_repeated_words(user_voice_output_raw)
    user_voice_output = remove_repetitive_phrases(user_voice_output)
    try:
        os.remove(WAVE_OUTPUT_FILENAME)
    except Exception as e:
        print("Error:", e)
        user_voice_output = ""
        return user_voice_output
    
    if len(user_voice_output.split()) <= 3:
        user_voice_output = ""
    
    # Print the recognized text
    print(user_voice_output)
    return user_voice_output
