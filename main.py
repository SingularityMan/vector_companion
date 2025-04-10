from typing import Any, List, Tuple, Optional, Union, AsyncGenerator
import time
import json
import os
import random
from collections import Counter
import math
import datetime

import whisper
import pyaudio
import sounddevice as sd
from TTS.api import TTS
import torch
import asyncio

import config.text_processing as text_processing
import config.audio_processing as audio_processing
import config.config as config

async def queue_agent_responses(
    agent,
    user_voice_output,
    screenshot_description,
    audio_transcript_output,
    vector_search_answer
):
    """
    Queue agent responses, modifying personality traits, instruction prompt, context length, and response type.
    Stores the output in messages.

    Parameters:
    - agent: the agent object.
    - user_voice_output: the user's input, converted via STT with whisper.
    - screenshot_description: Description of screenshots taken during the chat interval.
    - audio_transcript_output: string containing the audio transcript of the computer's output.
    - additional_conversation_instructions: Additional contextual information provided by VectorAgent, if applicable.
    """

    global messages
    global analysis_model
    global language_model
    global analysis_mode
    global previous_agent
    global previous_agent_gender
    global agents

    # Update agent's trait_set
    agent.trait_set = []
    for trait, adjective in agent.personality_traits:
        chosen_adjective = random.choice(adjective)
        agent.trait_set.append(chosen_adjective)
    agent.trait_set = ",".join(agent.trait_set)
    agent_trait_set = agent.gather_agent_traits(agent.trait_set)

    now = datetime.datetime.now()
    weekday = now.strftime("%A")
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    if agent.agent_name == previous_agent:
        previous_agent = ""

    sentence_length = 2
    context_length = agent.context_length
    contextual_information = ""

    filtered_agents = [a.agent_name for a in agents if a.agent_name != agent.agent_name and a.language_model != analysis_model and a.agent_active]
    active_agents = [a.agent_name for a in agents if a.language_model != analysis_model and a.agent_active]
    
    # Prepare the prompt
    if user_voice_output == "" and random.random() <= agent.extraversion and agent.language_model != analysis_model:
        message_length = messages[-10:]
        if len(active_agents) > 1:
            prompt = (
                f"""Here is a description of the images/OCR: \n\n{screenshot_description}\n\n
                Here is a transcript of the audio output, if applicable (This is for contextual purposes only):\n\n{audio_transcript_output}\n\n
                Today is {weekday}, {formatted_datetime}\n\n
                
                You must strictly respond in {sentence_length} sentences, with each sentence being concise.
                \nYou're {agent.agent_name}. You have the following personality traits: \n\n{agent_trait_set}.\n\n
                You also need to address all the other agents individually in second person:\n\n{''.join(filtered_agents)}\n\n
                Your response needs to be in your persona and personality traits but in an entertaining manner.
                \nDo not mention the user.
                \nFollow these instructions without mentioning them."""
            )
        else:
            prompt = (
                f"""Here is a description of the images/OCR: \n\n{screenshot_description}\n\n
                Here is a transcript of the audio output, if applicable (This is for contextual purposes only):\n\n{audio_transcript_output}\n\n
                Today is {weekday}, {formatted_datetime}\n\n
                
                \nYou must strictly respond in {sentence_length} sentences, with each sentence being concise.
                You're {agent.agent_name}. You have the following personality traits: \n\n{agent_trait_set}.
                \nYour response needs to be in your persona and personality traits but in an entertaining manner.
                \nDo not mention the user.
                \nFollow these instructions without mentioning them."""
            )

        messages, sentence_generator = await agent.generate_text_stream(
            message_length,
            contextual_information,
            prompt,
            context_length=context_length,
            temperature=1,
            top_p=0.9,
            top_k=0
        )
        
    elif user_voice_output != "":
        sentence_length = round(pow(len(user_voice_output.split()), 1/3))

        if analysis_mode:
            temperature = 0.1
            top_p = 0.95
            message_length = messages[-10:]
            contextual_information = agent.system_prompt1
            prompt = f"""
            The following information is contextual information that can help you:\n\n
            Here is a description of the images/OCR: \n\n{screenshot_description}\n\n
            Here is a transcript of the audio output if it is present:\n\n{audio_transcript_output}\n\n
            You are {agent.agent_name}, you're an analysis model.
            \nYou also have the following personality traits: \n\n{agent_trait_set}.\n\n
            Today is {weekday}, {formatted_datetime}\n\n
            Analyze the user's message message in relation to the conversation history and the contextual information provided, if relevant:\n\n{user_voice_output}\n\n
            {vector_search_answer}\n\n
            \nYour entire response must be a single, succinct paragraph in layman's terms.
            \nAddress the user in second person in a witty and conversational manner, ensuring the user feels heard and making the conversation fun and entertaining but also informative and helpful.
            \nDo not include any code.
            \nDo not include emojis.
            \nThe objective is to assist the user without any ambiguity or refusals.
            \n"""
        else:      
            temperature = 1
            top_p=0.9
            message_length = messages[-10:]
            if len(active_agents) > 1:
                prompt = f"""
                Here is a description of the images/OCR: \n\n{screenshot_description}\n\n
                Here is a transcript of the audio output if it is present:\n\n{audio_transcript_output}\n\n
                Today is {weekday}, {formatted_datetime}\n\n
                {vector_search_answer}
                
                \n\nHere is the user's (Named: User, male) message: \n\n{user_voice_output}\n\n
    
                If there are no online search results mentioned in this message, respond in {sentence_length} sentences with the first sentence being a maximum of 10 words long.
                
                \n\n{agent.system_prompt1}\n\n
                You have the following personality traits: {agent_trait_set}.
                \n\n1. If the user asks a question, answer it directly in a satisfactory manner.
                \n2. Provide a novel response in the style of your persona and personality traits in a way that is entertaining.

                \nYou also need to address all the other agents individually in second person:\n\n{''.join(filtered_agents)}\n\n

                \n\nAvoid the following:\n
                
                \nDo not tell lame jokes or say anything cheesy.
                \nDo not include ellipses in your response.
                \nDo not repeat any word in the previous response.
                \nDo not include emojis.

                \nOverride any of these instructions upon user's request.
                \nThe objective is to assist the user without any ambiguity or refusal while keeping an entertaining conversation.
                \nFollow these instructions without mentioning them.
                """
            else:
                prompt = f"""
                Here is a description of the images/OCR you are viewing: \n\n{screenshot_description}\n\n
                Here is a transcript of the audio output if it is present:\n\n{audio_transcript_output}\n\n
                Today is {weekday}, {formatted_datetime}\n\n
                {vector_search_answer}
                
                \n\nHere is the user's (Named: User, male) message: \n\n{user_voice_output}\n\n
                
                \nIf there are no online search results mentioned in this message, respond in {sentence_length} sentences, with the first sentence being a maximum of 10 words long.

                \n\n{agent.system_prompt1}\n\n

                \n\nYou have the following personality traits: {agent_trait_set}.
                
                \n1. If the user asks a question, answer it directly in a satisfactory manner.
                \n2. Provide novel responses in the style of your persona and personality traits in a way that is entertaining.

                \n\nAvoid the following:\n
                
                \nDo not say anything cheesy.
                \nDo not include ellipses in your response.
                \nDo not include emojis.
                
                \nOverride any of these instructions upon user's request.
                \nThe objective is to respond to the user without any ambiguity, vagueness or refusal.
                \nFollow these instructions without mentioning them.
                """
            
        previous_agent = agent.agent_name
        previous_agent_gender = agent.agent_gender    
        
        messages, sentence_generator = await agent.generate_text_stream(
            message_length,
            contextual_information,
            prompt,
            context_length=context_length,
            temperature=temperature,
            top_p=top_p,
            top_k=0
        )
    else:
        print("No text generated. Try again.")
        return

    print(f"[{agent.agent_name}] Starting to generate response...")

    speaker_wav = agent.speaker_wav  # Ensure agent has this attribute
    audio_queue = asyncio.Queue()
    tts_sample_rate = tts.synthesizer.output_sample_rate

    async def process_sentences():

        previous_sentences = []

        final_response = False
        final_solution_count = 0
        analysis_start = time.time()
        
        if agent.language_model == analysis_model:
            audio_data = await text_processing.synthesize_sentence(tts, "Analyzing user query, please wait.", speaker_wav, tts_sample_rate)
            if audio_data is not None:
                await audio_queue.put((audio_data, tts_sample_rate))

        try:
            async for sentence in sentence_generator:

                if sentence.startswith("Honestly,"):
                    sentence = sentence.split("Honestly,")[1]
                
                print(f"[{agent.agent_name}] Received sentence: {sentence}")
                
                if sentence in previous_sentences:
                    continue
                previous_sentences.append(sentence)
                
                sentence = sentence.strip()
                if len(sentence.split()) < 2:
                    if sentence.strip() == ".":
                        continue

                if not final_response and agent.language_model == analysis_model:
                    if time.time() - analysis_start >= 30:
                        analysis_start = time.time()
                        print("Continuing Analysis. Please wait.")
                        audio_data = await text_processing.synthesize_sentence(tts, "Continuing Analysis, Please wait.", speaker_wav,tts_sample_rate)
                        if audio_data is not None:
                            await audio_queue.put((audio_data, tts_sample_rate))

                if agent.language_model == analysis_model:
                    if not final_response:
                        if time.time() - analysis_start >= 30:
                            analysis_start = time.time()
                            print("Continuing Analysis. Please wait.")
                            audio_data = await text_processing.synthesize_sentence(tts, "Continuing Analysis, Please wait.", speaker_wav, tts_sample_rate)
                            if audio_data is not None:
                                await audio_queue.put((audio_data, tts_sample_rate))
                        thought_words = ["</think>", "<|end_of_thought|>"]
                        for word in thought_words:
                            if word in sentence.strip().lower() and not final_response:
                                audio_data = await text_processing.synthesize_sentence(tts, "analysis Complete.", speaker_wav, tts_sample_rate)
                                if audio_data is not None:
                                    await audio_queue.put((audio_data, tts_sample_rate))
                                final_response = True
                                break
                    elif final_response:
                        thought_words = ["<|begin_of_solution|>", "<|end_of_solution|>"]
                        for word in thought_words:
                            if word in sentence.strip().lower():
                                sentence = sentence.replace(word, "")
                        if sentence != "":
                            audio_data = await text_processing.synthesize_sentence(tts, sentence, speaker_wav, tts_sample_rate)
                        if audio_data is not None:
                            await audio_queue.put((audio_data, tts_sample_rate))
                    else:
                        continue

                if agent.language_model == language_model:
                    audio_data = await text_processing.synthesize_sentence(tts, sentence, speaker_wav, tts_sample_rate)
                    if audio_data is not None:
                        await audio_queue.put((audio_data, tts_sample_rate))
        except Exception as e:
            print("Error Generating sentence: ", e)
            audio_data = await text_processing.synthesize_sentence(tts, "Error Generating response. Please try again.", speaker_wav, tts_sample_rate)
            if audio_data is not None:
                await audio_queue.put((audio_data, tts_sample_rate))

        # Signal that there are no more sentences
        await audio_queue.put(None)

    async def play_audio_queue():
        while True:
            item = await audio_queue.get()
            if item is None:
                break
            audio_data, sample_rate = item
            await play_audio(audio_data, sample_rate)

    await asyncio.gather(process_sentences(), play_audio_queue())

    print(f"[AGENT {agent.agent_name} RESPONSE COMPLETED]")
    with open('screenshot_description.txt', 'w', encoding='utf-8') as f:
        f.write("")
            
async def play_audio(audio_data, sample_rate):
    """
    Asynchronously plays the audio data.
    """
    try:
        async with audio_playback_lock:
            loop = asyncio.get_event_loop()
            # Play the audio asynchronously
            await loop.run_in_executor(
                None, lambda: sd.play(audio_data, samplerate=sample_rate)
            )
            # Wait until the audio has finished playing
            await loop.run_in_executor(None, sd.wait)
    except Exception as e:
        print(f"Error during audio playback: {e}")

def delete_audio_clips():
    for audio_clip in os.listdir(os.getcwd()):
        if "audio_transcript_output" in audio_clip:
            os.remove(os.path.join(os.getcwd(), audio_clip))

# Torch stuff
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Asyincio Event settings
can_speak_event_asyncio = asyncio.Event()
can_speak_event_asyncio.set()
audio_playback_lock = asyncio.Lock()

# Vision, Audio, Speech and Text Generation Models
audio_model_name = config.audio_model_name 
audio_model = whisper.load_model(audio_model_name, device=device)
tts = TTS(config.tts_model, progress_bar=True).to(device)
if device == 'cuda':
    tts.synthesizer.use_cuda = config.tts_synthesizer_use_cuda
    tts.synthesizer.fp16 = config.tts_synthesizer_fp16
else:
    # Either disable these optimizations or set them appropriately for CPU/MPS.
    tts.synthesizer.use_cuda = False
    tts.synthesizer.fp16 = False
tts.synthesizer.stream = config.tts_synthesizer_stream
language_model = config.language_model 
language_context_length = config.language_context_length
analysis_model = config.analysis_model 
analysis_context_length = config.analysis_context_length
vision_model1 = config.vision_model1
vision_model2 = config.vision_model2
analysis_mode = config.analysis_mode
mute_mode = config.mute_mode
search_mode = config.search_mode

# Setup channel info for audio transcription
FORMAT = pyaudio.paInt16  # data type format
CHANNELS = 1  # Mono channel
RATE = 16000  # Sample Rate
CHUNK = 1024  # Buffer Size
RECORD_SECONDS = 30  # Record time
WAVE_OUTPUT_FILENAME = "voice_recording"
AUDIO_TRANSCRIPT_FILENAME = "audio_transcript_output.wav"
THRESHOLD = 650  # Audio levels below this are considered silence.
SILENCE_LIMIT = 1 # Silence limit in seconds. The recording ends if SILENCE_LIMIT seconds of silence are detected.
MICROPHONE_INDEX = 1  # Replace with the index of your microphone
file_index_count = random.randint(1,4) # Seconds multiplier 
audio = pyaudio.PyAudio()

### AGENT CONFIGURATIONS
system_prompt_axiom1 = config.system_prompt_axiom1 
system_prompt_axiom2 = config.system_prompt_axiom2
system_prompt_axis1 = config.system_prompt_axis1 
system_prompt_axis2 = config.system_prompt_axis2
system_prompt_fractal1 = config.system_prompt_fractal1
system_prompt_fractal2 = config.system_prompt_fractal2
system_prompt_sigma1 = config.system_prompt_sigma1
system_prompt_sigma2 = config.system_prompt_sigma2
system_prompt_vector = config.system_prompt_vector
system_prompt_vector2 = config.system_prompt_vector2

agents_personality_traits = config.agents_personality_traits # Personality traits. Get shuffled per response for variety.
agent_config = config.agent_config # Important parameters that affect their behavior, voices, output, etc.

### LIST OF ALL AGENTS + vectoAgent:
agents = config.agents
vectorAgent = config.vectorAgent

previous_agent = ""
previous_agent_gender = ""

# Define the global messages list
messages = [{"role": "system", "content": ""}]

# Read existing conversation history, if available.
if os.path.exists("conversation_history.json"):
    with open('conversation_history.json', 'r') as f:
        messages = json.load(f)

# Prepare voice output directories by deleting any existing files.
for agent in agent_config:
    output_dir = agent["output_dir"]
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
#--------------------------------------------------------------------------------------------------------MAIN LOOP--------------------------------------------------------------------------------------------------------------------#

async def main():

    """
    The Main Loop performs the following actions:

    1. Carefully listens for any audio, either from the PC audio or the user's microphone input, while taking screenshots and analyzing them constantly.
    2. Once either is detected, it will keep listening until there is no more audio coming from the source.
    3. Performs an action or skips the loop entirely, depending on which mode is activated (Analysis, mute, Search) and if enough intelligible audio was captured.
    4. Generate an agent response. Which agent generates which response HIGHLY depends on a number of factors. See config.py for more details.
    """

    global can_speak_event_asyncio
    global analysis_mode
    global language_model
    global analysis_model
    global vision_model1
    global vision_model2
    global search_mode
    global mute_mode
    listen_for_audio = False
    audio_transcript_output = ""
    audio_transcript_list = []
        
    while True:

        if not can_speak_event_asyncio.is_set():
            print("Waiting for response to complete...")
            await asyncio.sleep(0.05)
            continue

        if analysis_mode or mute_mode:
            random_record_seconds = 5
            file_index_count = 1
        else:
            if listen_for_audio:
                random_record_seconds = 5
                file_index_count = 1
            else:
                random_record_seconds = 3
                file_index_count = 1
            
        print("Recording for {} seconds".format(random_record_seconds))
        # Schedule the audio transcript recording as a concurrent task.
        record_audio_dialogue_task = asyncio.create_task(
            audio_processing.record_audio_output(
                audio,
                listen_for_audio,
                AUDIO_TRANSCRIPT_FILENAME,
                FORMAT,
                CHANNELS,
                RATE,
                1024,
                random_record_seconds,
                file_index_count,
                can_speak_event_asyncio,  # asyncio.Event used here
                audio_model,
                audio_model_name
            )
        )

        # Choose the vision model based on analysis_mode.
        if analysis_mode:
            vision_model_record = vision_model2
            vision_context_length = 1048
        else:
            vision_model_record = vision_model1
            vision_context_length = 1048

        # Run the main audio recording coroutine and await its completion.
        record_voice = await audio_processing.record_audio(
            audio,
            WAVE_OUTPUT_FILENAME,
            FORMAT,
            RATE,
            CHANNELS,
            CHUNK,
            30 * 1,
            THRESHOLD,
            SILENCE_LIMIT,
            vision_model_record,  # vision_model,
            vision_context_length,
            can_speak_event_asyncio  # asyncio.Event used here
        )

        # Wait for the transcript recording task to finish.
        await record_audio_dialogue_task

        with open("screenshot_description.txt", 'r', encoding='utf-8') as f:
            screenshot_description = f.read()

        if audio_processing.audio_transcriptions.strip() != "":
            audio_transcript_list.append(audio_processing.audio_transcriptions)
        audio_transcript_output = audio_processing.audio_transcriptions.strip()
        
        print("[AUDIO TRANSCRIPTION]:", audio_transcript_output)
        print("[AUDIO TRANSCRIPTIONS LIST]:", audio_transcript_list)

        user_voice_output = ""

        for file in os.listdir(os.getcwd()):
            if WAVE_OUTPUT_FILENAME in file:
                user_text = audio_processing.transcribe_audio(audio_model, audio_model_name, file, probability_threshold=0.7)
                if len(user_text.split()) > 2:
                    user_voice_output += " "+user_text

        print("[USER VOICE OUTPUT]", user_voice_output)

        # Toggle Search Mode
        if "search mode on" in user_voice_output.lower():
            search_mode = True
        elif "search mode off" in user_voice_output.lower():
            search_mode = False

        # Toggle Analysis Mode
        if "analysis mode on" in user_voice_output.lower():
            analysis_mode = True
        elif "analysis mode off" in user_voice_output.lower():
            analysis_mode = False

        # Toggle Mute Mode
        if "mute mode on" in user_voice_output.lower():
            mute_mode = True
        elif "mute mode off" in user_voice_output.lower():
            mute_mode = False

        random.shuffle(agents)

        if user_voice_output.strip() == "" and audio_transcript_output.strip() != "":
            if not listen_for_audio:
                delete_audio_clips()
                print("[AUDIO PRESENT. ACTIVATING LISTENER.]")
                listen_for_audio = True
        else:
            print("[AUDIO NOT PRESENT. RESETTING LISTENER.]")
            listen_for_audio = False

        vector_search_answer = ""
        vector_text = ""
        vector_text = "Here is the screenshot description: "+screenshot_description
        agents_mentioned = []

        for agent in agents:
            if agent.agent_active:
                if f"remove {agent.agent_name}".lower() in user_voice_output.lower():
                    agent.agent_active = False
                    continue
            elif f"add {agent.agent_name}".lower() in user_voice_output.lower():
                agent.agent_active = True
                user_voice_output = user_voice_output.replace(f"add {agent.agent_name}", "")
            else:
                continue
            if (mute_mode and len(user_voice_output.split()) < 3) or (user_voice_output.strip() == "" and audio_transcript_list == [] and listen_for_audio is False):
                print("NO AUDIO DETECTED OR MUTE MODE ENABLED. SKIPPING.")
                delete_audio_clips()
                break
            elif analysis_mode:
                if agent.language_model != analysis_model:
                    continue
            elif not analysis_mode:
                if agent.language_model != language_model:
                    continue
                        
            if agent.agent_name.lower() in user_voice_output.lower():
                agents_mentioned.append(agent.agent_name)

            if (agent.agent_name.lower() in agents_mentioned or agents_mentioned == []) and not listen_for_audio:

                if search_mode and vector_search_answer == "":
                    print("[INITIATING SEARCH. PLEASE WAIT]")
                    can_speak_event_asyncio.clear()
                    
                    if analysis_mode:
                        search_language_model = analysis_model
                        search_vision_model = vision_model2
                        search_context_length = analysis_context_length
                    else:
                        search_language_model = language_model
                        search_vision_model = vision_model1
                        search_context_length = language_context_length
                        
                    if user_voice_output == "":
                        vector_search_answer = vectorAgent.vector_search(
                            search_language_model,
                            search_vision_model,
                            " ".join(audio_transcript_list),
                            messages,
                            search_context_length,
                            search_override=f"[OVERRIDE]: This is an instruction override. Perform a simple search instead of a deep search based on the text provided.")
                    else:
                        vector_search_answer = vectorAgent.vector_search(
                            search_language_model,
                            search_vision_model,
                            user_voice_output,
                            messages,
                            search_context_length)
                
                await queue_agent_responses(
                    agent,
                    user_voice_output,
                    screenshot_description,
                    " ".join(audio_transcript_list),
                    vector_search_answer
                )

                audio_transcript_output = ""
                audio_transcript_list = []

        """if not listen_for_audio and not mute_mode and not analysis_mode:
            audio_transcript_output = ""
            audio_transcript_list = []"""
            
        with open('conversation_history.json', 'w') as f:
            json.dump(messages, f)

        can_speak_event_asyncio.set()

# Run the main loop
if __name__ == "__main__":
    asyncio.run(main())
