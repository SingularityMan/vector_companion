import whisper
import pyaudio
import wave
import audioop
import time
import requests
import json
import simpleaudio as sa
import subprocess
import threading
import base64
import pyautogui as pygi
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import re
import random
from collections import Counter
import time
from transformers import pipeline
from bs4 import BeautifulSoup
import librosa
import noisereduce as nr
import soundfile as sf
from datetime import datetime

# Get the directory that contains config.py
config_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to can_speak.txt
can_speak_path = os.path.join(config_dir, '..', 'can_speak.txt')

# Construct the paths to the text files
dialogue_axiom_path = os.path.join(config_dir, '..',  'dialogue_text_axiom.txt')
dialogue_axis_path = os.path.join(config_dir, '..',  'dialogue_text_axis.txt')

can_speak = True
can_speak_event = threading.Event()
can_speak_event.set()

#------------------------------------------TEXT PROCESSING-----------------------------------------------------------#

class Agent():

    def __init__(self, agent_name, agent_gender, personality_traits, system_prompt1, system_prompt2, dialogue_list):
        self.agent_name = agent_name
        self.agent_gender = agent_gender
        self.system_prompt1 = system_prompt1
        self.system_prompt2 = system_prompt2
        self.previous_agent_message = ""
        self.personality_traits = personality_traits
        self.trait_set = []
        self.dialogue_list = dialogue_list

    def summarize_conversation(self, messages, agent_messages):
        url = "http://localhost:11434/api/chat"
        headers = {
            "Content-Type": "application/json"
        }
        
        summary_prompt = "Provide an expository summary of this conversation in list format, highlighting the most significant events that occurred while being as objective as possible:\n\n" + "\n".join(agent_messages[-8000:])

        summary_payload = {
            "model": "llama3.1:8b-instruct-fp16",
            "messages": [{"role": "system", "content": "Summarize the conversation in list format."}, {"role": "user", "content": summary_prompt}],
            "stream": False,
            "options": {
                "repeat_penalty": 1.15,
                "num_ctx": 32000
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
        
    def generate_text(self, messages, agent_messages, system_prompt, user_input, temperature=0.7, top_p=0.3, top_k=2000):

        global image_lock

        # Define the endpoint and headers
        url = "http://localhost:11434/api/chat"
        headers = {
            "Content-Type": "application/json"
        }

        if len(messages) > 100:
            messages = [{"role": "system", "content": system_prompt}]
            agent_messages, conversation_summary = self.summarize_conversation(messages, agent_messages)
            messages.append({"role": "user", "content": conversation_summary})
            print("[CONVERSATION SUMMARY]:", conversation_summary)

        messages[0] = {"role": "system", "content": system_prompt}
            
        # Add the new user input to the messages list
        messages.append({"role": "user", "content": user_input})

        # Define the payload
        payload = {
            "model": "llama3.1:8b-instruct-fp16",
            "messages": messages,
            "stream": False,
            "options":{
                "repeat_penalty": 1.40,
                #"temperature": 1,
                #"top_p": top_p,
                #"top_k": top_k,
                "num_ctx": 32000
                #"stop": []
                }
        }

        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Check the response
        if response.status_code == 200:
            response_data = response.json()
            # Extract the assistant's message content
            text_response = response_data.get("message", {}).get("content", "No response received")
            if text_response.startswith('"') and text_response.endswith('"'):
                text_response = text_response[1:-2]
            # Remove unwanted characters
            text_response = re.sub(r'[^\x00-\x7F]+', '', text_response)
            # Remove text within parentheses
            text_response = re.sub(r'\(.*?\)', '', text_response)
            # Include asterisks and text between them
            text_response = re.sub(r'\*.*?\*', '', text_response)
            # Include visible "\n" in text outputs
            text_response = text_response.replace('\\n', '')
            #print(f"{self.agent_name} Response:", f"{text_response}")
            agent_messages.append(f"Agent Name:{self.agent_name} ({self.agent_gender})\n Agent Response: {text_response}")
            return messages, agent_messages, text_response
        else:
            print("Failed to get a response. Status code:", response.status_code)
            print("Response:", response.text)
            return messages, agent_messages, "Failed to get a response."

class VectorAgent():

    def __init__(self):
        self.objectives = []
        pass

    def gather_agent_traits(self, agent_traits):
        return ' '.join(agent_traits)

    def generate_text(self, agent_name, agent_messages, agent_traits, screenshot_description, audio_transcript_output):

        # Define the endpoint and headers
        url = "http://localhost:11434/api/chat"
        headers = {
            "Content-Type": "application/json"
        }

        agent_messages = ([{"role": "assistant", "content": ' '.join(agent_messages[-8000:])}])

        current_time = datetime.now().time()
            
        # Add the new user input to the messages list
        agent_messages.append({"role": "user", "content":
                               "Review this contextual information:"
                               "\n\nScreenshot\OCR description: "+screenshot_description+
                               "\n\nAudio Transcript Output: "+audio_transcript_output+

                               "\n\nYour task will be separated between three different categories, the scope ranging from broad to narrow for each task:"

                               "\n\n[TASK 1, BROADEST SCOPE, CONTEXT-ORIENTED]: Your first task will be to generate a detailed, one paragraph description of the current situation based on this context, highlighting the most important parts of the most recent situation while ignoring the lesser parts."
                               "\nThe sentence needs to place a special emphasis on the event that is occurring right now so the agents can remain up to date."

                               "\n\n[TASK 2, SECOND-MOST BROAD, OBJECTIVE-ORIENTED]: Your second task would be to set a one-sentence objective behind the scenes that augment the user's experience throughout the conversation."
                               "\nYou will use the contextual information provided and the converesation history in order to plan and execute an objective that is directly tied to them."
                               "\nThen use the agent in order to complete this objective."
                               "\nYou must update the objective in real-time, placing more emphasis on the most recent events and less emphasis on less recent ones."
                               "\nIn your objective statement, you must explain how this augments the user's experience."
                               "\nThe objective statement needs to be different from the instructions below. Those are geared towards the individual agent's behavior."
                               
                               "\n\n[TASK 3, NARROW SCOPE, AGENT-ORIENTED]: Your Final task will be to generate a single example response for the AI agent named "+agent_name+" containing the following qualities:"
                               
                               "\n\n1. The objective is to regulate and steer a conversation between two agents, named Axiom and Axis, matching the tone of both the most immediate context and the current agent's personality traits: "+agent_traits+" while preventing any kind of repetition by the agent."
                               "\n2. Each agent response needs to be completely in character, keeping the agent traits mentioned above. The most important part of the response is the agent's personality traits expressed by the agent and avoiding repetition, keeping the conversation fresh."
                               "\n3. Each agent response needs to be composed of 2 extremely brief, concise and contextually relevant sentences made up of more than 5 words but less than 10 words per sentence embodying the agent's unique personality traits."
                               "\n4. Each agent response needs to be completely different but true to the agent's character, avoiding similarities in previous responses. However, the context should be quickly updated based on the most immediate situation."
                               "\n5. If the agents start arguing with each other, break up the fight and keep their focus on the context."
                               "\n6. Disrupt up any unwanted patterns such as: arguing, repetition, irrelevance, breaking character, inattentiveness, incoherence, etc."
                               "\n7. You need to emphasize the agent's personality traits, the conversation history and the contextual information provided (focusing on key events and individuals in the context), combining all three to generate a response."
                               "\n8. The agent should also quickly shift the conversation in a different direction and avoid dwelling on past subjects for too long."
                               "\n9. The agent cannot speak poetically, metaphorically, nor use complex language and parables. The agent should be simple and direct yet be true to his character traits."
                               "\n10. You must penalize any repetition or similarity between agents, making it clear each sentence needs to be unique."
                               "\n11. The agent's response should be meaningful, not talking about mundane things like walls, trees, rocks, etc."
                               "\n12. The agent should not be asking questions. Instead, the agent should be making comments, statements and opinions based on their personality traits outlined above."
                               "\n13. If there is no audio transcript output, instruct the agent to focus on the immediate surroundings provided by the context instead."
                               "\n\nAll of these instructions for Task 3 need to be encapsulated in a single example sentence for the agent in question."
                               "\nThis means you're not supposed to outline these instructions, but instead provide an example response for the agent based on these instructions and personality traits of the agent."
                               
                               "\n\nComplete these tasks in order without mentioning them in any way. No acknowledgement, no offer of assistance, nothing. Just do it."})

        # Define the payload
        payload = {
            "model": "llama3.1:8b-instruct-fp16",
            "messages": agent_messages,
            "stream": False,
            "options":{
                #"repeat_penalty": 1.20,
                #"temperature": temperature,
                #"top_p": top_p,
                #"top_k": top_k,
                "num_ctx": 32000
                #"num_predict": 250,
                #"stop": []
                }
        }

        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Check the response
        if response.status_code == 200:
            response_data = response.json()
            # Extract the assistant's message content
            text_response = response_data.get("message", {}).get("content", "No response received")
            if text_response.startswith('"') and text_response.endswith('"'):
                text_response = text_response[1:-2]
            # Remove unwanted characters
            text_response = re.sub(r'[^\x00-\x7F]+', '', text_response)
            # Remove text within parentheses
            text_response = re.sub(r'\(.*?\)', '', text_response)
            #print(f"{self.agent_name} Response:", f"{text_response}")
            print("[VECTOR INSTRUCTIONS]:", text_response)
            return text_response
        else:
            print("Failed to get a response. Status code:", response.status_code)
            print("Response:", response.text)
            return "Failed to get a response."


def find_repeated_words(text, threshold=6):
    # Regular expression pattern to match words
    pattern = r'\b(\w+)\b'
    # Find all words
    words = re.findall(pattern, text, re.IGNORECASE)
    # Count occurrences of each word
    word_counts = Counter(words)
    # Filter out words repeated 6 or more times
    filtered_words = [word for word in words if word_counts[word.lower()] < threshold]
    # Clear the Counter object
    word_counts.clear()
    return " ".join(filtered_words)

def remove_repetitive_phrases(text):
    # Regular expression to match repeated sequences of words
    pattern = re.compile(r'(\b\w+\b(?:\s+\b\w+\b){0,4})(?:\s+\1)+', re.IGNORECASE)
    result = pattern.sub(r'\1', text)
    return result

def check_sentence_length(text, message_length=45, sentence_length=2):
    
    # Clean up output
    text = remove_repetitive_phrases(text)

    # Split sentences, but ignore ellipsis
    sentences = re.split(r'(?:(?<=[.?;]))\s', text.strip())
    #print("Sentences: ", sentences)
    if sentences:
        if len(sentences) >= sentence_length:
            print("Sentence 1 length: ", len(sentences[0]))
            print("Sentence 2 length: ", len(sentences[1]))
            return (sentences[:sentence_length], ' '.join(sentences[:sentence_length]))
        else:
            return sentences, sentences[0]
    return "No valid text found."

def remove_repetitive_phrases(text, max_repeats=3):
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
def view_image(vision_model, processor):

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
            max_new_tokens=150,
            do_sample=False,
            num_beams=1
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
    except:
        pass

    #print(parsed_answer)

def view_image_ocr(vision_model, processor):

    try:

        prompt = "<OCR>"

        image_file = "axiom_screenshot.png"
        image = Image.open(image_file)

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = vision_model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=500,
            do_sample=True,
            temperature=1,
            num_beams=10
        )

        generated_ids.to('cpu')
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task='<OCR>', 
            image_size=(image.width, image.height)
        )

        #print("<OCR>: ", parsed_answer)

        current_time = datetime.now().time()

        with open("screenshot_description.txt", "a", encoding='utf-8') as f:
                f.write(f"\n\nOCR text in image at {current_time.strftime('%H:%M:%S')}:\n\n"+parsed_answer['<OCR>'])
    except:
        pass

#-------------------------------------------------AUDIO PROCESSING---------------------------------------------#

def record_audio(audio, WAVE_OUTPUT_FILENAME, FORMAT, RATE, CHANNELS, CHUNK, RECORD_SECONDS, THRESHOLD, SILENCE_LIMIT, vision_model, processor):

    global image_lock
    global can_speak
    
    ii = 0
    
    try:
        while True:

            # Cancel recording if Agent speaking
            if not can_speak_event.is_set():
                time.sleep(1)
                print("[record_user_mic] Waiting for response to complete...")
                continue
            
            # Start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=1, frames_per_buffer=CHUNK)
            #print("waiting for speech...")
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
                #print(f"Current RMS: {rms}")  # Debugging RMS values
                if ii < int(RATE / CHUNK * RECORD_SECONDS):
                    ii += 1
                    #print("FRAMES MICROPHONE INPUT: "+str(ii)+"/"+str(int(RATE / CHUNK * RECORD_SECONDS)))
                    if ii % (int(RATE / CHUNK) * 10) * 10 == 0:
                        print("CHECKPOINT------------------------",ii)
                        image_picture = pygi.screenshot("axiom_screenshot.png")
                        if not image_lock:
                            threading.Thread(target=view_image, args=(vision_model, processor)).start()

                if rms > THRESHOLD and not recording_started:
                    SILENCE_LIMIT = 1
                if ii >= int(RATE / CHUNK * RECORD_SECONDS) and not recording_started:
                    SILENCE_LIMIT = 1

                if rms > THRESHOLD or (ii >= int(RATE / CHUNK * RECORD_SECONDS) and not recording_started):
                    if not recording_started:
                        print("recording...")
                        image_picture = pygi.screenshot("axiom_screenshot.png")
                        if not image_lock:
                            threading.Thread(target=view_image, args=(vision_model, processor)).start()
                        recording_started = True
                    frames.append(data)
                    THRESHOLD = 300
                    silence_start = time.time()  # reset silence timer
                elif recording_started:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_LIMIT:
                        print("finished recording")
                        can_speak_event.clear()
                        break
                else:
                    pass
                    #print(f"rms: {rms}, threshold: {THRESHOLD}")

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

def record_audio_output(audio, WAVE_OUTPUT_FILENAME, FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS):

    global can_speak

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
                        input_device_index=2)

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
        can_speak = False
        
        if not can_speak_event.is_set():

            # Stop the stream
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save the audio to a .wav file
            wf = wave.open('audio_transcript_output.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            break

        frames = []

def transcribe_audio(model, WAVE_OUTPUT_FILENAME, RATE=44100):

    # Load audio and pad/trim it to fit 60 seconds
    audio_data = whisper.load_audio(WAVE_OUTPUT_FILENAME)
    audio_data = whisper.pad_or_trim(audio_data)

    # Ensure audio data is of the correct shape
    max_audio_length = 60 * RATE  # 60 seconds * sample rate
    audio_data = audio_data[:max_audio_length]

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio_data).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # English language only supported
    if 'en' not in detected_language:
        return ""

    # Decode the audio
    options = whisper.DecodingOptions(
    task="transcribe",
    length_penalty=1,  # Apply length penalty for better long transcriptions
    prompt=None,
    prefix=None,
    suppress_tokens="-1",
    suppress_blank=True,
    without_timestamps=True,  # Disable timestamps if not needed
    max_initial_timestamp=1.0,
    fp16=True  # Keep using fp16 for performance
    )

    result = whisper.decode(model, mel, options)
    user_voice_output = result.text
    #user_voice_output = find_repeated_words(user_voice_output_raw)
    #user_voice_output = remove_repetitive_phrases(user_voice_output)
    os.remove(WAVE_OUTPUT_FILENAME)
    
    if len(user_voice_output.split()) <= 3:
        user_voice_output = ""
    
    # Print the recognized text
    print(user_voice_output)
    return user_voice_output
