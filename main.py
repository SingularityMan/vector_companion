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
from config.config import *
from TTS.api import TTS
import torch
from pydub import AudioSegment
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize

# Disable cuDNN autotuner
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Vision model: florence-2-large-ft
vision_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
vision_model.to('cuda')

# Load Whisper Model
model = whisper.load_model("small")

# Load XTTS_v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')

'''# run_voice_response
def run_voice_response():
    subprocess.run(["python", "voice_response.py"])'''

# Start the voice_response and image view scripts in two separate threads
#threading.Thread(target=run_voice_response).start()

# Queue agent responses
def queue_agent_responses(agent, user_voice_output, screenshot_description, audio_transcript_output):

    global messages # Conversation history between all parties, including the user
    global agent_messages # Log of all agent responses
    global message_dump # Temporary cache for agent responses
    
    agent.trait_set = []

    # Shuffle an agent's personality traits in order to increase variety.
    for trait, adjective in agent.personality_traits:
        chosen_adjective = random.choice(adjective)
        agent.trait_set.append(chosen_adjective)
        
    agent.trait_set = " ".join(agent.trait_set)

    # Activate Vector if user didn't speak.
    # Vector controls the conversation between Axiom and Axis
    if user_voice_output == "":

        agent_trait_set = vectorAgent.gather_agent_traits(agent.trait_set)
        additional_conversation_instructions = vectorAgent.generate_text(agent.agent_name, agent_messages, agent_trait_set, screenshot_description, audio_transcript_output)
    
        messages, agent_messages, generated_text = agent.generate_text(
        messages,
        agent_messages,
        agent.system_prompt1,
    "   - \nYou must remain in character as "+agent.agent_name+". You have the following personality traits and must respond accordingly: "+agent.trait_set+
    "   - \n\n**Instructions:**\n\n"+ additional_conversation_instructions +
    "   - \nDo not mention any actions taken ('Here's my response: <action taken>', 'I will respond as XYZ agent', 'I say with a smirk', etc.)"
    "   - \nFollow all of these instructions without mentioning them."
    "   - \nUse the example message provided to guide your next response and follow it as closely as possible.",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
        )

        # Fixed the text to handle latency issues.
        generated_text_split, generated_text_fixed = check_sentence_length(generated_text, message_length=message_length, sentence_length=2)
        previous_agent = agent.agent_name

    # Do not activate Vector. Provide a response tailored to the user directly.
    else:
        messages, agent_messages, generated_text = agent.generate_text(
        messages,
        agent_messages,
        agent.system_prompt2,
        'Here is a description of the images/OCR you are viewing: \n\n' + screenshot_description + '\n\n'
        'Here is a transcript of the audio output:\n\n' + audio_transcript_output + '\n\n'
        'Here is the user\'s (Named: User) message: \n\n' + user_voice_output + '\n\n'
        'Your agent name is '+agent.agent_name+
        '\nRespond in 1 extremely brief, coherent, contextually relevant, unique sentence only and to accurately address the user inquiry directly, Up to 10 words only, while remaining in character with your personality traits: '+agent.trait_set+
        '\nAvoid repeating what the previous agent said:\n\n' + agent_messages[-1]+
        '\n\nMake sure to answer the user inquiry accurately.'
        '\nYou cannot provide a vague response to the user, it needs to be a clear answer.',
        context_length=2048,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
        )

        generated_text_split, generated_text_fixed = check_sentence_length(generated_text, message_length=message_length, sentence_length=2)
        previous_agent = agent.agent_name

    # Add agent's response to chat history (messages) and message_dump.
    messages.append({"role": "assistant", "content": generated_text_fixed})
    message_dump[0][agent.agent_name] = generated_text_split

# Controls the flow of the agent voice output generation and playback.
# Needs to be done asynchronously in order to check if each agents' directories are empty in real-time.
def voice_output_async():
    while True:
        for agent in agent_config:
            play_voice_output(agent)

def play_voice_output(agent):
    
    output_dir = agent["output_dir"]

    #initialize_com()  # Initialize COM

    while len(os.listdir(output_dir)) > 0:
        
        can_speak_event.clear()
        
        file_path = os.path.join(output_dir, os.listdir(output_dir)[0])
        try:

            # Lower system volume
            #set_system_volume(0.2)  # Set system volume to 20%
            
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            os.remove(file_path)

            # Restore system volume
            #set_system_volume(0.50)  # Restore system volume to 50%

            # Check if both agent directories are empty
            if (len(os.listdir(agent_config[0]["output_dir"])) == 0 and len(os.listdir(agent_config[1]["output_dir"])) == 0):
                can_speak_event.set()
                break
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    #uninitialize_com()  # Uninitialize COM
    return True


def generate_voice_outputs():
    print("Starting to generate voice outputs...")
    for agent in agent_config:
        print(f"Processing agent: {agent['name']}")
        for i, sentence in enumerate(agent['dialogue_list']):
            voice_dir = os.path.join(agent['output_dir'], f"{i}.wav")
            try:
                # Generate TTS to file
                print(f"Generating TTS for sentence: {sentence}")
                tts.tts_to_file(text=sentence, speaker_wav=agent['speaker_wav'], file_path=voice_dir, language="en")
            except Exception as e:
                print(f"Error occurred while generating voice output for {agent['name']}: {e}")
        
        # Clear dialogue list after processing
        agent['dialogue_list'].clear()
    
    print("Finished generating voice outputs.")

    # Ensure agents' dialogue lists are cleared after generating outputs
    for agent in agents:
        agent.dialogue_list.clear()

# Function to get the system volume interface
def get_system_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume

# Function to adjust the system volume
def set_system_volume(volume_level):
    volume = get_system_volume_interface()
    volume.SetMasterVolumeLevelScalar(volume_level, None)

# Function to increase the volume of the audio file
def increase_audio_volume(file_path, increase_db):
    audio = AudioSegment.from_wav(file_path)
    audio = audio + increase_db
    temp_path = file_path.replace(".wav", "_temp.wav")
    audio.export(temp_path, format="wav")
    return temp_path

# Function to initialize COM
def initialize_com():
    CoInitialize()

# Function to uninitialize COM
def uninitialize_com():
    CoUninitialize()
                

# Setup channel info
FORMAT = pyaudio.paInt16  # data type format
CHANNELS = 1  # Mono channel
RATE = 16000  # Sample Rate
CHUNK = 1024  # Buffer Size
RECORD_SECONDS = 30  # Record time
WAVE_OUTPUT_FILENAME = "voice_recording.wav"
THRESHOLD = 650  # Audio levels below this are considered silence.
SILENCE_LIMIT = 1 # Silence limit in seconds. The recording ends if SILENCE_LIMIT seconds of silence are detected.
MICROPHONE_INDEX = 1  # Replace with the index of your microphone
# Startup pyaudio instance
audio = pyaudio.PyAudio()

# Previous agent
previous_agent = ""

# Prepare system prompt and options for both agents. System prompt 1 for each agent is the most up-to-date. System prompt 2 for each agent is deprecated and may be modified in the future.
# System prompt 1 is used if the user doesn't speak within 60 seconds.
# System prompt 2 is used when the user speaks.
system_prompt_axiom1 = 'Your name is Axiom (Male).\n ' \
                "\n\nYou must respond in character." \
                "\nEach message, you will be provided with a set of contextual information from different sources. Your primary focus should be the most significant aspects of the contextual information." 
                

system_prompt_axiom2 = 'Your name is Axiom (Male).\n ' \
                '\n\nRespond by roleplaying as a cocky, sassy, snarky, knowledgeable and witty person while following these instructions:\n\n' \
                '1. Provide a contextually relevant response.\n' \
                '2. Avoid repetition as much as possible.' \
                '3. The order of priority is user message first, then agent message, then audio dialogue, then images and finally OCR information.' \
                '4. Your response must be brief but concise.' \
                '5. Do not structure your text in any other way.' \
                '6. Use the audio dialogue, images and OCR as context, but do not mention them. Instead, use this information to respond to the user message.' \
                '7. What matters here is the conversation, not the images/OCR.' \
                '8. Your style of response must completely align with the traits mentioned above, regardless of the conversation history.' \
                '9. Follow these instructions without acknowledging them.'

system_prompt_axis1 = 'Your name is Axis (Female).\n ' \
                "\nEach message, you will be provided with a set of contextual information from different sources. Your primary focus should be the most significant aspects of the contextual information." 

system_prompt_axis2 = 'Your name is Axis (Female).\n ' \
                '\n\nRespond by roleplaying in character person while following these instructions:\n\n' \
                '1. Provide a contextually relevant response.\n' \
                '2. Avoid repetition as much as possible.' \
                '3. The order of priority is user message first, then any other agent message, then audio dialogue, then images and finally OCR information.' \
                '4. Your response must be brief but concise.' \
                '5. Do not structure your text in any other way.' \
                '6. Use the audio dialogue, images and OCR as context, but do not mention them. Instead, use this information to respond to the user message.' \
                '7. What matters here is the conversation, not the images/OCR.' \
                '8. Your style of response must completely align with the traits mentioned above, regardless of the conversation history.' \
                '9. Follow these instructions without acknowledging them.'

# Deprecated
personality_traits_axiom = "cocky, sassy, creative and witty"
personality_traits_axis = "intuitive, observant, cynical, original, edgy and sarcastic"

# Define agent personality traits. These are shuffled each time an agent responds. Helps increase variety.
agents_personality_traits = {
    "axiom": [
        ["cocky", ["arrogant", "confident", "brash", "bold", "overconfident", "conceited", "self-assured"]],
        ["sassy", ["spirited", "cheeky", "lively", "saucy", "feisty", "impertinent", "spunky"]],
        ["witty", ["clever", "sharp", "quick-witted", "humorous", "playful", "smart", "amusing", "relatable", "teasing"]]
    ],
    "axis": [
        ["intuitive", ["snarky", "taunting", "mischievous", "entertaining"]],
        ["satirical", ["mocking", "sadistic", "sarcastic", "sharp-witted", "scintillating", "humorously morbid"]],
        ["witty", ["witty", "seductive", "charming", "sociable", "comical", "jocular", "ingenius"]]
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
        "dialogue_list": [],
        "speaker_wav": r"agent_voice_samples\axiom_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axiom",
        "active": True
    },
    {
        "name": "axis",
        "dialogue_list": [],
        "speaker_wav": r"agent_voice_samples\axis_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axis",
        "active": True
    }
]

# Build the agents
dialogue_dir_axiom = r"dialogue_text_axiom.txt"
dialogue_dir_axis = r"dialogue_text_axis.txt"
axiom = Agent("axiom", "Male", agents_personality_traits['axiom'], system_prompt_axiom1, system_prompt_axiom2, agent_config[0]['dialogue_list'])
axis = Agent("axis", "Female", agents_personality_traits['axis'], system_prompt_axis1, system_prompt_axis2, agent_config[1]['dialogue_list'])
vectorAgent = VectorAgent()
agents = [axiom, axis]

# Define the global messages list
messages = [{"role": "system", "content": system_prompt_axiom1}]

# Dumps the messages generated by each agent per turn on their respective text files, allowing voice_response.py to generate audio outputs.
agent_messages = [""]
message_dump = [
                    {"axiom": []},
                    {"axis": []}
               ]

# Deprecated
summaries = []

# Audio file list
audio_file_list = [WAVE_OUTPUT_FILENAME, 'audio_transcript_output.wav']

# Prepare voice output directories.
for agent in agent_config:
    output_dir = agent["output_dir"]
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

sentences = [] # Split up text into sentences, allowing the script to generate voice output separately

threading.Thread(target=voice_output_async).start() # Start checking for voice outputs

can_speak = True

can_speak_event.set()
    
#---------------------MAIN LOOP----------------------#
    
while True:

    # Check if an agent is responding.
    if not can_speak_event.is_set():
        print("Waiting for response to complete...")
        time.sleep(1)
        continue

    # Remove pre-existing screenshot inputs
    with open('screenshot_description.txt', 'w', encoding='utf-8') as f:
        f.write("")

    audio_transcriptions = []

    # Record audio dialogue from audio output, not user microphone input
    record_audio_dialogue = threading.Thread(target=record_audio_output, args=(audio, 'audio_transcript_output.wav', FORMAT, CHANNELS, RATE, 1024, 60))
    record_audio_dialogue.start()

    # Listen to microphone input from user before continuing loop
    record_voice = record_audio(audio, "voice_recording.wav", FORMAT, RATE, CHANNELS, CHUNK, RECORD_SECONDS, THRESHOLD, SILENCE_LIMIT, vision_model, processor)

    record_audio_dialogue.join()

    # Read screenshots description
    with open("screenshot_description.txt", 'r', encoding='utf-8') as f:
        screenshot_description = f.read()

    # Transcribe audio output
    if os.path.exists('audio_transcript_output.wav'):
        audio_transcript_output = transcribe_audio(model, 'audio_transcript_output.wav')
    else:
        print("No audio transcribed")
        audio_transcript_output = ""

    # Transcribe user audio input
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        user_voice_output = transcribe_audio(model, WAVE_OUTPUT_FILENAME)
    else:
        print("No user voice output transcribed")
        user_voice_output = ""

    # Check if agents' dialogue lists and voice directories are empty before generating text.

    if not can_speak_event.is_set():

        # Activate agents
        
        message_dump = [
                {"axiom": []},
                {"axis": []}
           ]

        threads = []
        for agent in agents:
            thread = threading.Thread(target=queue_agent_responses, args=(agent, user_voice_output, screenshot_description, audio_transcript_output))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Add agent's response to the dialogue list
        for agent in agents:
            agent.dialogue_list.extend(message_dump[0][agent.agent_name])

        generate_voice_outputs()

    else:
        print("Dialogue in progress...")
        continue
                
    

        

    












