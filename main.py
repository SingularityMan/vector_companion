import time
import requests
import json
import subprocess
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import base64
import os
import re
import random
from collections import Counter
import math
from ctypes import cast, POINTER

from PIL import Image
import whisper
import pyaudio
import wave
import audioop
import simpleaudio as sa
from transformers import AutoProcessor, AutoModelForCausalLM
from TTS.api import TTS
import torch
from pydub import AudioSegment
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize

from config.config import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
lock = Lock()

# Vision, Audio, Speech and Text Generation Models
vision_path = r"microsoft/Florence-2-large-ft"
vision_model = AutoModelForCausalLM.from_pretrained(vision_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(vision_path, trust_remote_code=True)
vision_model.to('cuda')
model = whisper.load_model("base")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to('cuda')
language_model = "llama3.1:8b-instruct-fp16"


def queue_agent_responses(agent: str, user_voice_output: str, screenshot_description: str, audio_transcript_output: str, additional_conversation_instructions: str):

    """
    Queue agent responses, modifying personality traits, instruction prompt, context length and response type.
    Stores the output in messages, agent messages and message dump.

    Parameters:
    - agent: the name of the agent.
    
    - user_voice_output: the user's input, converted via STT with whisper.
    
    - screenshot_description: Description of all the screenshots taken during the chat interval, stored in screenshot_description.txt.
    
    - audio_transcript_output: string containing the audio transcript of the computer's output, converted via STT with whisper.
    
    - additional_conversation_instructions: Additional contextual information provided by VectorAgent, if applicable.

    """

    global messages 
    global agent_messages 
    global message_dump 
    global user_memory 
    
    agent.trait_set = []

    for trait, adjective in agent.personality_traits:
        chosen_adjective = random.choice(adjective)
        agent.trait_set.append(chosen_adjective)
        
    agent.trait_set = ", ".join(agent.trait_set)
    agent_trait_set = vectorAgent.gather_agent_traits(agent.trait_set)

    if user_voice_output == "":

        sentence_length = 3

        agent_prompt_list = [

                    "You're " + agent.agent_name + ". You are a human who is "+ agent.trait_set +"."
                    "\n\nIn a maximum of " + str(sentence_length) + " short sentences less than 15 words long, respond with a focus the current situation."
                    "\nKeep your replies concise and direct. Prioritize clarity and avoid over-elaboration."
                    "\nYour ultimate goal is ensuring a satisfying and human-like interaction.."
                    "\nYour goal is to NOT be agreeable and obedient, but have your own identity while responding like "+ agent.trait_set +". You need to have interesting responses."
                    "\nRemember to follow the rule of cool."
                    "\nFollow these instructions without mentioning them."
                ]

        agent_prompt = random.choice(agent_prompt_list)
        context_length = (len(audio_transcript_output.split())*3)+(len(additional_conversation_instructions.split())*3)+len(screenshot_description.split())*3
    
        messages, agent_messages, generated_text = agent.generate_text(
            messages,
            agent_messages,
            agent.system_prompt1,
            "\n\nHere is a transcript of the audio: \n\n'"+audio_transcript_output+"'\n\nAdditional contextual information: "+additional_conversation_instructions+"\n\n"+agent_prompt,
            context_length=context_length,
            temperature=random.random(),
            top_p=random.random(),
            top_k=0
        )

    else:

        sentence_length = round(pow(len(user_voice_output.split()), 1/3))
        if sentence_length > 4:
            sentence_length = 4
        context_length = (len(user_voice_output.split())*100)
            
        messages, agent_messages, generated_text = agent.generate_text(
        messages,
        agent_messages,
        agent.system_prompt2,
        "Here is a description of the images/OCR you are viewing: \n\n" + screenshot_description + "\n\n"
        "Here is a transcript of the audio output:\n\n" + audio_transcript_output + "\n\n"
        "Here is the user's (Named: User, male) message: \n\n" + user_voice_output + "\n\n"
        "You may only use the most relevant details about the following user's traits if applicable:\n\n"+'\n'.join(user_memory)+"\n\n"
        "You are "+agent.agent_name+". You are a human who is "+ agent.trait_set +"."
        "\nRespond in "+str(sentence_length)+" contextually relevant sentences, with each sentence being no more than"+ str(len(user_voice_output.split()) // 2) +
        "words long, only addressing the user inquiry directly and expanding on the previous agent's message."
        "\nKeep your replies concise and direct. Prioritize clarity and avoid over-elaboration."
        "\nYour ultimate goal is ensuring a satisfying and human-like interaction."
        "\nYour goal is to NOT be agreeable and obedient, but have your own identity. You need to have interesting responses."
        "\nRemember to follow the rule of cool."
        "\nFollow these instructions without mentioning them.",
        context_length=context_length,
        temperature=random.random(),
        top_p=random.random(),
        top_k=0
        )

    # Fixed the text to handle latency issues.
    generated_text_split, generated_text_fixed = check_sentence_length(generated_text, sentence_length=sentence_length)
    print("[AGENT "+agent.agent_name+" RESPONSE]:", generated_text_fixed)

    # Add agent's response to chat history (messages) and message_dump.
    messages.append({"role": "assistant", "content": generated_text_fixed})
    message_dump[0][agent.agent_name] = generated_text_split

def voice_output_async():

    """

    Controls the flow of the agent voice output generation and playback.
    Needs to be done asynchronously in order to check if each agents' directories are empty in real-time.
    
    """
    
    while True:
        for agent in agent_config:
            play_voice_output(agent)

def play_voice_output(agent: str) -> bool:

    """
    Play audio file of assigned agent.

    Disables user speaking, plays audio files and once files on both agent folders are clear,
    enables user speaking.
    
    Returns a boolean.

    Parameter:

    agent: specified agent

    """
    
    output_dir = agent["output_dir"]

    while len(os.listdir(output_dir)) > 0:
        
        can_speak_event.clear()
        
        file_path = os.path.join(output_dir, os.listdir(output_dir)[0])
        try:
            
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            os.remove(file_path)

            if (len(os.listdir(agent_config[0]["output_dir"])) == 0 and len(os.listdir(agent_config[1]["output_dir"])) == 0):
                can_speak_event.set()
                break
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    return True


def generate_voice_outputs(agent):

    """
    Uses xttsv2 to generate agent audio.
    Generates the audio sentence-by-sentence per each agent's dialogue_list.
    Each sentence contains its own audio file and is stored in each agent's output_dir.
    
    """
    """print("Starting to generate voice outputs...")
    for agent in agent_config:"""
    
    with lock:
        print(f"Processing agent: {agent['name']}")
        for i, sentence in enumerate(agent['dialogue_list']):
            voice_dir = os.path.join(agent['output_dir'], f"{i}.wav")
            try:
                print(f"Generating TTS for sentence: {sentence}")
                tts.tts_to_file(text=sentence, speaker_wav=agent['speaker_wav'], file_path=voice_dir, language="en")
            except Exception as e:
                print(f"Error occurred while generating voice output for {agent['name']}: {e}")
        agent['dialogue_list'].clear()
    print("Finished generating voice outputs.")
                
# Setup channel info
FORMAT = pyaudio.paInt16  # data type format
CHANNELS = 1  # Mono channel
RATE = 16000  # Sample Rate
CHUNK = 1024  # Buffer Size
RECORD_SECONDS = 30  # Record time
WAVE_OUTPUT_FILENAME = "voice_recording.wav"
AUDIO_TRANSCRIPT_FILENAME = "audio_transcript_output.wav"
THRESHOLD = 650  # Audio levels below this are considered silence.
SILENCE_LIMIT = 1 # Silence limit in seconds. The recording ends if SILENCE_LIMIT seconds of silence are detected.
MICROPHONE_INDEX = 1  # Replace with the index of your microphone
file_index_count = 2 # Seconds multiplier 
audio = pyaudio.PyAudio()

# Reset prompts. Newer models don't follow system prompts.
system_prompt_axiom1 = 'Your name is Axiom (Male).\n ' 
system_prompt_axiom2 = 'Your name is Axiom (Male).\n '
system_prompt_axis1 = 'Your name is Axis (Female).\n ' 
system_prompt_axis2 = 'Your name is Axis (Female).\n ' 

"""
Define agent personality traits.
These are shuffled each time an agent responds.
Helps increase variety.
You can add and remove as many categories and traits as you like.
"""

agents_personality_traits = {
    "axiom": [
        ["cocky", ["cocky"]],
        ["sassy", ["witty"]],
        ["witty", ["chatty"]],
        ["entertaining", ["entertaining"]]
    ],
    "axis": [
        ["intuitive", ["intuitive"]],
        ["satirical", ["sarcastic"]],
        ["witty", ["conversational"]],
        ["entertaining", ["entertaining"]]
    ]
}

# Deprecated
temperature = 0.3
top_p = 0.3
top_k=2000

sentence_length = 2 # Truncates the message to 2 sentences per response
message_length = 45 # Deprecated

# Agent configurations
agent_config = [
    {
        "name": "axiom",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\axiom_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axiom",
        "active": True
    },
    {
        "name": "axis",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\axis_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axis",
        "active": True
    }
]

# Build the agents
dialogue_dir_axiom = r"dialogue_text_axiom.txt"
dialogue_dir_axis = r"dialogue_text_axis.txt"
axiom = Agent("axiom", "Male", agents_personality_traits['axiom'], system_prompt_axiom1, system_prompt_axiom2, agent_config[0]['dialogue_list'], language_model)
axis = Agent("axis", "Female", agents_personality_traits['axis'], system_prompt_axis1, system_prompt_axis2, agent_config[1]['dialogue_list'], language_model)
vectorAgent = VectorAgent(language_model)
agents = [axiom, axis]

# Define the global messages list
messages = [{"role": "system", "content": system_prompt_axiom1}]

if os.path.exists("conversation_history.json"):
    # Read existing history
    with open('conversation_history.json', 'r') as f:
        messages = json.load(f)

agent_messages = [message["content"] for message in messages if message.get("role") == "assistant"]
if len(agent_messages) == 0:
    agent_messages = [""]

# Memory feature. Agent remembers User's personality traits.
if os.path.exists("user_memory.json"):
    with open("user_memory.json", 'r') as f:
        user_memory = json.load(f)
    for memory in user_memory:
        print(memory)
else:
    user_memory = [""]

message_dump = [
                    {"axiom": []},
                    {"axis": []}
               ]

# Prepare voice output directories by deleting any existing files.
for agent in agent_config:
    output_dir = agent["output_dir"]
    
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        
        if os.path.isfile(file_path):
            os.remove(file_path)


threading.Thread(target=voice_output_async).start()
sentences = []
can_speak = True
can_speak_event.set()
    
#---------------------MAIN LOOP----------------------#

"""
The Main Loop performs the following actions:

1. Check if the user can speak. Otherwise, it will wait for the agents to finish speaking.
2. Reset screenshot_description, audio_transcript_output and user_voice_output.
3. Starts recording for 60 seconds and takes/analyzes screenshots periodically. Stops after 60 seconds or user begins speaking.
4. Prompts VectorAgent to generate a description of the situation if necessary, then prompts the agents to generate their own responses.
5. Voice output is played after agents finish generating their dialogue.
"""
    
while True:

    if not can_speak_event.is_set():
        print("Waiting for response to complete...")
        time.sleep(0.05)
        continue

    with open('screenshot_description.txt', 'w', encoding='utf-8') as f:
        f.write("")

    audio_transcriptions = ""

    record_audio_dialogue = threading.Thread(target=record_audio_output, args=(audio, AUDIO_TRANSCRIPT_FILENAME, FORMAT, CHANNELS, RATE, 1024, 30, file_index_count))
    record_audio_dialogue.start()

    record_voice = record_audio(audio, WAVE_OUTPUT_FILENAME, FORMAT, RATE, CHANNELS, CHUNK, RECORD_SECONDS*file_index_count, THRESHOLD, SILENCE_LIMIT, vision_model, processor)
    record_audio_dialogue.join()

    with open("screenshot_description.txt", 'r', encoding='utf-8') as f:
        screenshot_description = f.read()

    for file in os.listdir(os.getcwd()):
        if "audio_transcript_output" in file:
            file_path = os.path.join(os.getcwd(), file)
            if os.path.isfile(file_path):
                audio_transcript_output = transcribe_audio(model, file_path)
                audio_transcriptions += audio_transcript_output
            else:
                print("No audio transcribed")
                audio_transcriptions = ""

    audio_transcript_output = audio_transcriptions

    if os.path.exists(WAVE_OUTPUT_FILENAME):
        user_voice_output = transcribe_audio(model, WAVE_OUTPUT_FILENAME)

        vector_text = ""

        _, __, generated_text = agents[0].generate_text(
            messages,
            agent_messages,
            agents[0].system_prompt2,
            "Read this message and respond in 1 sentence detailing anything noteworthy about the user's personality and interests, if any:\n\n"+user_voice_output+"\n\n"
            "If nothing significant about the user is mentioned, only respond with a one-word 'None'."
            "Do not highlight your response."
            "Follow these instructions without mentioning them.",
            context_length=2048,
            temperature=0.1,
            top_p=0.1,
            top_k=0,
        )

        if len(generated_text.split()) > 1:
            user_memory.append(generated_text)

            if len(user_memory) > 10:
                user_memory.remove(user_memory[0])
        
    else:
        print("No user voice output transcribed")
        user_voice_output = ""

    if user_voice_output.strip() == "":
        vector_text = vectorAgent.generate_text(
            messages,
            screenshot_description,
            audio_transcript_output,
            context_length=2048
        )

    if not can_speak_event.is_set():
        
        message_dump = [
                {"axiom": []},
                {"axis": []}
           ]

        threads = []

        for i, agent in enumerate(agents):
            queue_agent_responses(agent, user_voice_output, screenshot_description, audio_transcript_output, vector_text)
            if message_dump[0].get(agent.agent_name):
                agent.dialogue_list.extend(message_dump[0][agent.agent_name])
            else:
                print(f"No messages found for {agent.agent_name}")
            #agent.dialogue_list.extend(message_dump[0][agent.agent_name])
            if len(messages) < 100:
                thread = threading.Thread(target=generate_voice_outputs, args=(agent_config[i],)).start()
                threads.append(thread)
            else:
                generate_voice_outputs(agent_config[i])

        for thread in threads:
            if thread != None:
                thread.join()
            
        with open('conversation_history.json', 'w') as f:
            json.dump(messages, f)

        with open('user_memory.json', 'w') as f:
            json.dump(user_memory, f)

        #generate_voice_outputs()

    else:
        print("Dialogue in progress...")
        continue
                
    

        

    












