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

# Vision model: florence-2-large-ft
vision_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
vision_model.to('cuda')

# Load Whisper Model
model = whisper.load_model("base")

# run_voice_response
def run_voice_response():
    subprocess.run(["python", "TTS\\voice_response.py"])

# Start the voice_response and image view scripts in two separate threads
threading.Thread(target=run_voice_response).start()

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
    "   - **Instructions to Follow:\n**"
    "   - \nYou must remain in character as "+agent.agent_name+". You have the following personality traits and must respon accordingly: "+agent.trait_set+
    "   - \n\n**Instructions:**\n\n"+additional_conversation_instructions +
    "   - \n\nDo not mention the contextual information provided."
    "   - \nDo not mention any actions taken ('Here's my response: <action taken>', 'I will respond as XYZ agent', etc."
    "   - \nFollow all of these instructions without mentioning them."
    "   - \nYour response needs to be completely different and unique every time, avoid any similarities to previous responses as much as possible."
    "   - \nDo not argue with the previous agent.",
        
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
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
        )

        generated_text_split, generated_text_fixed = check_sentence_length(generated_text, message_length=message_length, sentence_length=2)
        previous_agent = agent.agent_name

    # Add agent's response to chat history (messages) and message_dump.
    messages.append({"role": "assistant", "content": generated_text_fixed})
    message_dump[0][agent.agent_name] = generated_text_split
    

# Setup channel info
FORMAT = pyaudio.paInt16  # data type format
CHANNELS = 1  # Mono channel
RATE = 44100  # Sample Rate
CHUNK = 2048  # Buffer Size
RECORD_SECONDS = 60  # Record time
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
        ["intuitive", ["passive-aggressive", "condescending", "snarky", "taunting", "derisive", "mischievous"]],
        ["satirical", ["mocking", "acerbic", "sarcastic", "dry", "disdainful"]],
    ]
}


# Deprecated
temperature = 0.3
top_p = 0.3
top_k=2000

sentence_length = 2 # Truncates the message to 2 sentences per response
message_length = 45 # Deprecated

# Build the agents
dialogue_dir_axiom = r"dialogue_text_axiom.txt"
dialogue_dir_axis = r"dialogue_text_axis.txt"
axiom = Agent("axiom", "Male", agents_personality_traits['axiom'], system_prompt_axiom1, system_prompt_axiom2, dialogue_dir_axiom)
axis = Agent("axis", "Female", agents_personality_traits['axis'], system_prompt_axis1, system_prompt_axis2, dialogue_dir_axis)
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

# Initialize, Clear any pre-existing dialogue and enable the user to speak.
with open(dialogue_dir_axiom, 'w', encoding='utf-8') as f:
    f.write("")

with open(dialogue_dir_axis, 'w', encoding='utf-8') as f:
    f.write("")

with open(r"can_speak.txt", 'w', encoding='utf-8') as f:
    f.write("True")
    
#---------------------MAIN LOOP----------------------#
    
while True:

    # Check if an agent is responding.
    with open(r"can_speak.txt", 'r', encoding='utf-8') as f:
        if f.read().strip() != "True":
            print("Waiting for response to complete...")
            time.sleep(1)
            continue

    # Remove pre-existing screenshot inputs
    with open('screenshot_description.txt', 'w', encoding='utf-8') as f:
        f.write("")

    # Record audio dialogue from audio output, not user microphone input
    threading.Thread(target=record_audio_output, args=(audio, 'audio_transcript_output.wav', FORMAT, CHANNELS, RATE, CHUNK, 60)).start()

    # Listen to microphone input from user before continuing loop
    record_voice = record_audio(audio, "voice_recording.wav", FORMAT, RATE, CHANNELS, CHUNK, RECORD_SECONDS, THRESHOLD, SILENCE_LIMIT, vision_model, processor)

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

    # Check if agents' text directories are empty (no pending sentences to generate) before generating agent response.
    # Otherwise skip this entire sequence.
    with open(dialogue_dir_axiom, 'r', encoding='utf-8') as f:
        dialogue_axiom_text = f.read()

    with open(dialogue_dir_axis, 'r', encoding='utf-8') as f:
        dialogue_axis_text = f.read()

    if (dialogue_axiom_text == "" and dialogue_axis_text == "") and len(os.listdir(r"TTS\agent_voice_outputs\axiom")) == 0 and len(os.listdir(r"TTS\agent_voice_outputs\axis")) == 0:

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

        for agent in agents:
            # Add agent's response to text file so a voice response can be generated.
            with open(agent.dialogue_dir, 'w', encoding='utf-8') as f:
                f.write('\n'.join(message_dump[0][agent.agent_name]))

    else:
        print("Dialogue in progress...")
        continue
                
    

        

    












