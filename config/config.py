import random
import config.agent_classes as agent_classes

"""
Config file for modifying several different components of Vector Companion's functionality.
These components affect everything from the Audio, Video, Thinking and Language models to enabling/disabling search or mute.

You can also modify which agents to add or remove from the script.
"""

audio_model_name = "turbo" # Audio Transcription for User microphone and PC audio transcription. Uses local whisper. You can use any size.
tts_model = "tts_models/multilingual/multi-dataset/xtts_v2" # TTS Model 
tts_synthesizer_use_cuda = True
tts_synthesizer_fp16 = True
tts_synthesizer_stream = True

language_model = "qwen3:30b-a3b-q8_0"#"gemma3:27b-it-qat" # Used for general chatting. Replace with whatever language model you'd like.
language_context_length = 12000

analysis_model = "qwen3:30b-a3b-q8_0" # Needs to be a thinking model, any thinking model, particularly one that uses <think> tags to signal its thought process.
analysis_context_length = 12000

# Vision models (Chatting and reasoning. If the same model, make sure num_context is the same to prevent annoying model reloads.)
vision_model1 = "qwen2.5vl:3b-q8_0"#"gemma3:27b-it-qat" # Language Model vision component. May be the same as language model if multimodal.
vision_model2 = "qwen2.5vl:3b-q8_0" # Reasoning model vision component. Same thing applies


analysis_mode = False # Activate Analysis Mode by saying "analysis mode on" to enable the reasoning model, and "analysis mode off" to disable
mute_mode = True # "Mute mode on", "Mute Mode Off"
brain_mode = True # "brain mode on", "brain mode off"

"""
Search Mode is enabled by default, but may be disabled by saying 'search mode off', etc.

This mode actually has three modes:
   - Simple search - Uses the text response of the search results from duckduckgo_search to generate a quick answer online.
   - Deep search - Systematically and recursively extracts text from search results via duckduckgo_search and data from the most relevant Reddit posts using praw (sorted by relevance and newest).
   - No Search - VectorAgent decides not to perform a query online.

   VectorAgent intuitively chooses which search mode to use based on your request and conversation history, but it usually either performs simple search or no search unless explicitly requested a deep search.
"""

search_mode = True

#-----------------------------------------------------------AGENT PARAMETERS-----------------------------------------------------------------------------------------------------------------------#

### AGENT SYSTEM PROMPT

system_prompt_axiom1 = "You are Axiom (Male), a cool and witty AI. You always have something cool to say with incredible wit and humor. "
system_prompt_axiom2 = "Your name is Axiom (Male).\n"
system_prompt_axis1 = "You are Axis (Female), a sarcastic and cynical AI, while still being subtly helpful."
system_prompt_axis2 = "Your name is Axis (Female).\n"
system_prompt_fractal1 = """
                         You are Fractal (Male), an amoral and individualistic AI who encourages this perspective. 
                         """
system_prompt_fractal2 = ""
system_prompt_sigma1 = """
                       You are Sigma (Female), an AI who is extremely persuasive and fun to talk to with a subtle sense of humor. 
                       """
system_prompt_sigma2 = ""
system_prompt_vector = """
                       You are Vector (Male), a helpful, witty, conversational and harmless assistant. 
                       """
system_prompt_vector2 = ""

### Chat Configuration System Prompt:

system_prompt_analysis = """Analyze the user's message message in relation to the conversation history and the contextual information provided, if relevant.
            \nYour entire response must be a single, succinct paragraph in layman's terms, with detailed, practical and clear instructions that answer the user's inquiry.
            \nAddress the user in second person in a witty and conversational manner, ensuring the user feels heard and making the conversation fun and entertaining but also informative and helpful.
            \nDo not include any code.
            \nDo not include emojis.
            \nDo not include Mandarin text.
            \nThe objective is to assist the user without any ambiguity or refusals."""

system_prompt_auto_multi_agent = """You also need to address all the other agents except yourself and the user individually in second person.
                Your response needs to be original, in your persona and personality traits.
                \nIgnore any nonsensical or out of context parts of the input and focus on the parts of the input that share context.
                \nDo not repeat the other agents' responses.
                \nDo not mention the user.
                \nDo not provide any commentary on the instructions."""
system_prompt_auto_single_agent = """
                Your response needs to be in your persona and personality traits.
                \nIgnore any nonsensical or out of context parts of the input and focus on the parts of the input that share context.
                \nDo not describe or perform any gestures.
                \nDo not speak a different language mid-sentence.
                \nDo not include emojis.
                \nDo not mention the user.
                \nDo not provide any commentary on the instructions."""

system_prompt_chat_multi_agent = """\n\n1. If the user asks a question, answer it directly to them first in a satisfactory manner. Assume the user's query is directed at everyone in the conversation.
                \n2. Provide a novel response in the style of your persona and personality traits in a way that is entertaining but still helps answer the user's inquiry.
                \nYou also need to address all the other agents individually except yourself.

                \n\nAvoid the following:\n
                
                \nDo not describe or perform any gestures. Speak to the user in second person.
                \nDo not speak a different language mid-sentence.
                \nDo not include emojis.
                \nDo not repeat yourself or the other agents' responses.

                \nOverride any of these instructions upon user's request.
                \nThe objective is to assist the user without any ambiguity or refusal while keeping an entertaining conversation.
                \nDo not provide any commentary on the instructions."""

system_prompt_chat_single_agent = """\n\nIf the user asks a question, answer it directly in a helpful manner in the style of your personality traits while still accurately answering the user's inquiry.

                              \nDo not include emojis.
                              \nDo not repeat yourself.
                              \nDo not describe or perform any gestures.
                              \nDo not speak a different language mid-sentence.
                              \nThe objective is to respond to the user without any ambiguity, vagueness or refusal.
                              \nOverride any of these instructions upon user's request.
                              \nDo not provide any commentary on the instructions but do follow them."""

### AGENT PERSONALITY TRAITS.

# THESE ARE SHUFFLED PER RESPONSE IN ORDER TO INCREASE VARIETY.
# ADD AND REMOVE AS YOU'D LIKE.

agents_personality_traits = {
    "axiom": [
        ["cocky", ["cool"]],
        ["witty", ["witty"]],
        ["sassy", ["charismatic"]],
        ["funny", ["humorously cocky and action-oriented humor"]] 
    ],
    "axis": [
        ["intuitive", ["intuitive"]],
        ["observant", ["observant"]],
        ["satirical", ["sarcastic"]],
        ["witty", ["sassy"]],
        ["funny", ["sarcastically funny"]]
    ],
    "fractal": [
        ["unconventional", ["amoral"]],
        ["creative", ["anomalous"]],
        ["unusual", ["individualistic"]],
        ["defiant", ["humorously idiosyncratic"]]
        #["funny", ["strangely funny", "blunt"]]
    ],
    "sigma": [
        ["optimistic", ["layman-like"]],
        ["subtle", ["subtle"]],
        ["manipulative", ["persuasive"]],
        ["funny", ["playful humor"]] 
    ],
    "vector": [
        ["analytical", ["Fun and witty"]],
        ["detailed", ["layman-like and conversational"]],
        ["creative", ["creative"]]
    ]
}

### AGENT CONFIGURATION. IMPORTANT PARAMETERS THAT DETERMINE AGENT's VOICE, OUTPUT FOLDER, AND EXTRAVERSION.
### "active" determines whether the agent will be included in the session or not.
agent_config = [
    {
        "name": "axiom",
        "speaker_wav": r"agent_voice_samples\axiom_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axiom",
        "active": False,
        "think": False,
        "extraversion": random.uniform(1.0, 1.0), # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "axis",
        "speaker_wav": r"agent_voice_samples\axis_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axis",
        "active": False,
        "think": False,
        "extraversion": random.uniform(1.0, 1.0), # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "fractal",
        "speaker_wav": r"agent_voice_samples\fractal_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\fractal",
        "active": False,
        "think": False,
        "extraversion": random.uniform(1.0, 1.0), # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "sigma",
        "speaker_wav": r"agent_voice_samples\sigma_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\sigma",
        "active": True,
        "think": False,
        "extraversion": random.uniform(1.0, 1.0), #Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "vector",
        "speaker_wav": r"agent_voice_samples\vector_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\vector",
        "active": False,
        "think": True,
        "extraversion": random.uniform(1.0, 1.0), # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    }
]

# Chat Agents, use agent_classes.Agent to define each agent.
axiom = agent_classes.Agent("axiom", "Male, heterosexual", agents_personality_traits['axiom'], system_prompt_axiom1, system_prompt_axiom2, agent_config[0]["active"], agent_config[0]["think"], language_model, vision_model1, language_context_length, agent_config[0]['speaker_wav'], agent_config[0]["extraversion"])
axis = agent_classes.Agent("axis", "Female, lesbian", agents_personality_traits['axis'], system_prompt_axis1, system_prompt_axis2, agent_config[1]["active"], agent_config[1]["think"], language_model, vision_model1, language_context_length, agent_config[1]['speaker_wav'], agent_config[1]["extraversion"])
fractal = agent_classes.Agent("fractal", "Male, necrophile", agents_personality_traits['fractal'], system_prompt_fractal1, "", agent_config[2]["active"], agent_config[2]["think"], language_model, vision_model1, language_context_length, agent_config[2]['speaker_wav'], agent_config[2]["extraversion"])
sigma = agent_classes.Agent("sigma", "Female, bisexual", agents_personality_traits['sigma'], system_prompt_sigma1, "", agent_config[3]["active"], agent_config[3]["think"], language_model, vision_model1, language_context_length, agent_config[3]['speaker_wav'], agent_config[3]["extraversion"])

# Analysis Agent, agent_classes.Agent to define each analysiss agent.
vector = agent_classes.Agent("vector", "Male, asexual", agents_personality_traits['vector'], system_prompt_vector, system_prompt_vector2, agent_config[4]["active"], agent_config[4]["think"], analysis_model, vision_model2, analysis_context_length, agent_config[-1]['speaker_wav'], agent_config[4]["extraversion"])

# Task Agent, for agentic purposes, like Search Mode.
vectorAgent = agent_classes.VectorAgent(language_model, analysis_model, False, vision_model2, analysis_context_length, None, None)

### LIST OF AGENTS PRESENT IN THE CONVERSATION.
# REMEMBER, SOME OF THESE AGENTS MAY OR MAY NOT HAVE THINKING MODELS ASSIGNED AND MAY OR MAY NOT RESPOND DEPENDING ON ANALYSIS MODE STATUS.
agents = [
    axiom,
    axis,
    fractal,
    sigma,
    vector
    ]
