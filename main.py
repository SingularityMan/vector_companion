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
import math

# Disable cuDNN autotuner
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Vision model: florence-2-large-ft
vision_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
vision_model.to('cuda')

# Load Whisper Model
model = whisper.load_model("base")

# Load XTTS_v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to('cuda')

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
        humor_list = [
                    "Parodying",
                    "Lampooning",
                    "Mocking",
                    "Ridiculing",
                    "Caricaturing",
                    "Deriding",
                    "Spoofing",
                    "Burlesquing",
                    "Mimicking",
                    "Poking fun at",
                    "Roasting",
                    "self-deprecating"
                ]
        
        humor = random.choice(humor_list)
    
        messages, agent_messages, generated_text = agent.generate_text(
        messages,
        agent_messages,
        agent.system_prompt1,
    "   - \nYou are "+agent.agent_name+". You are a AI agent that essentially has the following personality traits: "+agent.trait_set+
    "   - \nHere is a transcript of the audio:\n\n"+ audio_transcript_output +
    "   - \n\n**Additional Context:**\n\n"+ additional_conversation_instructions +
    "   - \n\nDo not mention any actions taken ('Here's my response: <action taken>', 'I will respond as XYZ agent', 'I say with a smirk', etc.)"
    "   - \nYou must provide a brief, 2-sentence response with a special emphasis on the current situation and the previous agent's quoted message: \n\n'"+agent_messages[-1]+"'.\n"
    "   - \nMake sure to address the previous agent directly in a podcast-style response."
    "   - \nFollow all of these instructions without mentioning them.",
        context_length=(len(audio_transcript_output.split())*50)+(len(additional_conversation_instructions.split())*100),
        temperature=1,
        top_p=0.9,
        top_k=100000
        )

        # Fixed the text to handle latency issues.
        generated_text_split, generated_text_fixed = check_sentence_length(generated_text, message_length=message_length, sentence_length=2)
        previous_agent = agent.agent_name

    # Do not activate Vector. Provide a response tailored to the user directly.
    else:

        # Modify response parameters based on user input length
        sentence_length = round(math.cbrt(len(user_voice_output.split())))
        if sentence_length > 4:
            sentence_length = 4
        context_length = (len(user_voice_output.split())*100)
        if context_length > 8000:
            context_length = 8000

        agent_trait_set = vectorAgent.gather_agent_traits(agent.trait_set)
            
        messages, agent_messages, generated_text = agent.generate_text(
        messages,
        agent_messages,
        agent.system_prompt2,
        'Here is a description of the images/OCR you are viewing: \n\n' + screenshot_description + '\n\n'
        'Here is a transcript of the audio output:\n\n' + audio_transcript_output + '\n\n'
        'Here is the user\'s (Named: User, male) message: \n\n' + user_voice_output + '\n\n'
        '\nRespond in '+str(sentence_length)+' contextually relevant sentences, with each sentence being no more than'+ str(len(user_voice_output.split()) // 2) +
        'words long, only addressing the user inquiry directly with the following personality traits: '+agent.trait_set+''
        '\nYou are required to give clear, concise, helpful, practical advice when needed, applying genuine suggestions according to the current situation.'
        '\nFollow these instructions without mentioning them.',
        context_length=2048,
        temperature=0.7,
        top_p=top_p,
        top_k=10000
        )

        generated_text_split, generated_text_fixed = check_sentence_length(generated_text, message_length=message_length, sentence_length=sentence_length)
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
file_index_count = 2 # Seconds multiplier 

# Startup pyaudio instance
audio = pyaudio.PyAudio()

# Previous agent
previous_agent = ""

# Prepare system prompt and options for both agents. System prompt 1 for each agent is the most up-to-date. System prompt 2 for each agent is deprecated and may be modified in the future.
# System prompt 1 is used if the user doesn't speak within 60 seconds.
# System prompt 2 is used when the user speaks.
system_prompt_axiom1 = 'Your name is Axiom (Male).\n ' 
                
system_prompt_axiom2 = 'Your name is Axiom (Male).\n '

system_prompt_axis1 = 'Your name is Axis (Female).\n ' 

system_prompt_axis2 = 'Your name is Axis (Female).\n ' 

# Deprecated
personality_traits_axiom = "cocky, sassy, creative and witty"
personality_traits_axis = "intuitive, observant, cynical, original, edgy and sarcastic"

# Define agent personality traits. These are shuffled each time an agent responds. Helps increase variety.
agents_personality_traits = {
    "axiom": [
        ["cocky", ["arrogant", "confident", "brash", "bold", "overconfident", "conceited", "self-assured", "upbeat"]],
        ["sassy", ["spirited", "badass", "cheeky", "lively", "saucy", "feisty", "impertinent", "spunky"]],
        ["witty", ["clever", "sharp", "quick-witted", "humorous", "playful", "smart", "amusing", "relatable", "teasing"]]
    ],
    "axis": [
        ["intuitive", ["attentive", "observant", "intuitive", "insightful"]],
        ["satirical", ["mocking", "sadistic", "sarcastic", "sharp-witted", "scintillating", "humorously morbid", "badass"]],
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
axiom = Agent("axiom", "Male", agents_personality_traits['axiom'], system_prompt_axiom1, system_prompt_axiom2, agent_config[0]['dialogue_list'])
axis = Agent("axis", "Female", agents_personality_traits['axis'], system_prompt_axis1, system_prompt_axis2, agent_config[1]['dialogue_list'])
vectorAgent = VectorAgent()
agents = [axiom, axis]

# Define the global messages list
messages = [{"role": "system", "content": system_prompt_axiom1}]

if os.path.exists("conversation_history.json"):
    # Read existing history
    with open('conversation_history.json', 'r') as f:
        messages = json.load(f)
    for message in messages:
        print(message)

# Dumps the messages generated by each agent per turn on their respective text files, allowing voice_response.py to generate audio outputs.
agent_messages = [message["content"] for message in messages if message.get("role") == "assistant"]
if len(agent_messages) == 0:
    agent_messages = [""]
    
print("[AGENT MESSAGES]:", agent_messages)
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
        time.sleep(0.05)
        continue

    # Remove pre-existing screenshot inputs
    with open('screenshot_description.txt', 'w', encoding='utf-8') as f:
        f.write("")

    audio_transcriptions = ""

    # Record audio dialogue from audio output, not user microphone input
    record_audio_dialogue = threading.Thread(target=record_audio_output, args=(audio, 'audio_transcript_output.wav', FORMAT, CHANNELS, RATE, 1024, 30, file_index_count))
    record_audio_dialogue.start()

    # Listen to microphone input from user before continuing loop
    record_voice = record_audio(audio, "voice_recording.wav", FORMAT, RATE, CHANNELS, CHUNK, RECORD_SECONDS*file_index_count, THRESHOLD, SILENCE_LIMIT, vision_model, processor)

    record_audio_dialogue.join()

    # Read screenshots description
    with open("screenshot_description.txt", 'r', encoding='utf-8') as f:
        screenshot_description = f.read()

    # Transcribe audio output
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
    print("[AUDIO TRANSCRIPT OUTPUT]:", audio_transcript_output)

    """if audio_transcript_output.strip() == "":
        file_index_count = 2
    elif file_index_count < 6:
        file_index_count += 1"""

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
            queue_agent_responses(agent, user_voice_output, screenshot_description, audio_transcript_output)
            #thread = threading.Thread(target=queue_agent_responses, args=(agent, user_voice_output, screenshot_description, audio_transcript_output))
            #thread.start()
            #threads.append(thread)

        # Wait for all threads to complete
        """for thread in threads:
            thread.join()"""

        # Add agent's response to the dialogue list
        for agent in agents:
            agent.dialogue_list.extend(message_dump[0][agent.agent_name])

        # Write updated history back to file
        with open('conversation_history.json', 'w') as f:
            json.dump(messages, f)

        generate_voice_outputs()

        print("[CONVERSATION LENGTH]:", len(messages))

    else:
        print("Dialogue in progress...")
        continue
                
    

        

    












