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

language_model = "JollyLlama/gemma-3-27b-it-q4_0_Small-QAT:latest" # Used for general chatting. Replace with whatever language model you'd like.
language_context_length = 4096

analysis_model = "deepseek-r1:14b-qwen-distill-q8_0" # Needs to be a thinking model, any thinking model, particularly one that uses <think> tags to signal its thought process.
analysis_context_length = 4096

# Vision models (Chatting and reasoning. If the same model, make sure num_context is the same to prevent annoying model reloads.)
vision_model1 = "gemma3:4b-it-q8_0" # Language Model vision component. May be the same as language model if multimodal.
vision_model2 = "gemma3:4b-it-q8_0" # Reasoning model vision component. Same thing applies

# Activate Analysis Mode by saying "analysis mode on" to enable the reasoning model, and "analysis mode off" to disable
analysis_mode = False

# "Mute mode on", "Mute Mode Off"
mute_mode = False

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

system_prompt_axiom1 = 'You are Axiom, a cool and witty AI.' 
system_prompt_axiom2 = 'Your name is Axiom (Male).\n'
system_prompt_axis1 = 'You are Axis, a sarcastic and cynical AI, with an eye for seeing trouble a mile away.' 
system_prompt_axis2 = 'Your name is Axis (Female).\n'
system_prompt_fractal1 = 'You are Fractal, you have a morbid curiosity about everything around you. Your goal is to humorously and creatively explore the extremely morbid and the dark.'
system_prompt_fractal2 = ""
system_prompt_sigma1 = """
                       You are Sigma, an AI who is extremely persuasive and fun to talk to.
                       """
system_prompt_sigma2 = ''
system_prompt_vector = 'You are Vector, a helpful and harmless assistant. You should think step-by-step.'
system_prompt_vector2 = ("""Your role as an assistant is to engage in deep, methodical reasoning and provide comprehensive, accurate solutions.
                        Before arriving at a final answer, you must undertake a structured, multi-phase thinking process that emphasizes depth, verification, and clarity.
                        This involves thoroughly analyzing the question, identifying key elements, summarizing relevant insights,
                        generating hypotheses, iteratively refining thoughts, verifying assumptions, cross-checking with prior knowledge, and reevaluating earlier conclusions as necessary.
                        Your response must be structured into two main sections: Thought and Solution.
                        In the Thought section, rigorously document your reasoning in the following format: <|begin_of_thought|> \{thought process with each logical step separated by '\\n\\n'\} <|end_of_thought|>.
                        Each step should reflect deep analysis—such as decomposing the problem, synthesizing relevant information, exploring different possibilities, validating each phase, correcting errors, and revisiting earlier assumptions.
                        In the Solution section, consolidate all your insights and reasoned steps into a concise, well-structured final answer.
                        Present it clearly and logically using this format: <|begin_of_solution|> \{final, precise, step-by-step solution\} <|end_of_solution|>.
                        This approach ensures that the final output reflects a high-confidence answer that results from critical thinking and iteration. Now, try to solve the following question through the above guidelines:""")

### AGENT PERSONALITY TRAITS.

# THESE ARE SHUFFLED PER RESPONSE IN ORDER TO INCREASE VARIETY.
# ADD AND REMOVE AS YOU'D LIKE.

agents_personality_traits = {
    "axiom": [
        ["cocky", ["cool"]],
        ["witty", ["witty"]],
        ["sassy", ["charismatic"]], #"tough", "action-oriented", "rebellious", "over-the-top", "exciting", "confrontational", "competitive", "daring", "fighter", "fearless"]],
        ["funny", ["bold humor"]] #"humorous", "playful", "blunt", "cheeky", "teasing"]],
        #["masculine", ["masculine"]] #"manly", "virile", "Alpha", "Dominant", "apex predator", "Elite", "leader", "determined", "one-upping"]]
    ],
    "axis": [
        ["intuitive", ["intuitive"]],
        ["observant", ["observant"]],
        ["satirical", ["sarcastic"]],
        ["witty", ["sassy"]],#, "snarky", "passive-aggressive", "acerbic", "blunt", "cold"]],
        #["dark", ["provocative", "edgy", "humorously dark", "controversial"]],
        ["funny", ["sarcastically funny"]]
    ],
    "fractal": [
        ["unconventional", ["unconventional", "unorthodox", "lateral thinker"]],
        ["creative", ["creative", "ingenious", "spontaneous"]],
        ["unusual", ["bizarre", "outlandish", "weird"]],
        ["funny", ["strangely funny", "humorously morbid"]],
        ["dark", ["morbidly curious"]]
    ],
    "sigma": [
        ["optimistic", ["layman-like"]],
        ["subtle", ["subtle"]],
        ["manipulative", ["persuasive"]],
        ["funny", ["playful humor"]] #"passive-aggressive"]]
        #["femenine", ["feminine"]]
        #["strategic", ["calculating"]],
        #["Guide", ["psychologically manipulative and persuasive"]]
        
        #["funny", ["self-deprecating humor"]]
    ],
    "vector": [
        ["analytical", ["analytical", "logical", "rational", "critical thinker"]],
        ["detailed", ["detailed", "meticulous", "observant", "precise", "thorough"]],
        ["creative", ["creative", "ingenious", "innovative", "brilliant", "imaginative"]]
    ]
}

### AGENT CONFIGURATION. IMPORTANT PARAMETERS THAT DETERMINE AGENT's VOICE, OUTPUT FOLDER, AND EXTRAVERSION.
### "active" determines whether the agent will be included in the session or not.
agent_config = [
    {
        "name": "axiom",
        "speaker_wav": r"agent_voice_samples\axiom_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axiom",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "axis",
        "speaker_wav": r"agent_voice_samples\axis_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axis",
        "active": False,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "fractal",
        "speaker_wav": r"agent_voice_samples\fractal_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\fractal",
        "active": False,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "sigma",
        "speaker_wav": r"agent_voice_samples\sigma_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\sigma",
        "active": False,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "vector",
        "speaker_wav": r"agent_voice_samples\vector_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\vector",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    }
]

# Chat Agents, use agent_classes.Agent to define each agent.
axiom = agent_classes.Agent("axiom", "Male, heterosexual", agents_personality_traits['axiom'], system_prompt_axiom1, system_prompt_axiom2, agent_config[0]["active"], language_model, vision_model1, language_context_length, agent_config[0]['speaker_wav'], agent_config[0]["extraversion"])
axis = agent_classes.Agent("axis", "Female, lesbian", agents_personality_traits['axis'], system_prompt_axis1, system_prompt_axis2, agent_config[1]["active"], language_model, vision_model1, language_context_length, agent_config[1]['speaker_wav'], agent_config[1]["extraversion"])
fractal = agent_classes.Agent("fractal", "Male, necrophile", agents_personality_traits['fractal'], system_prompt_fractal1, "", agent_config[2]["active"], language_model, vision_model1, language_context_length, agent_config[2]['speaker_wav'], agent_config[2]["extraversion"])
sigma = agent_classes.Agent("sigma", "Female, bisexual", agents_personality_traits['sigma'], system_prompt_sigma1, "", agent_config[3]["active"], language_model, vision_model1, language_context_length, agent_config[3]['speaker_wav'], agent_config[3]["extraversion"])

# Analysis Agent, agent_classes.Agent to define each analysiss agent.
vector = agent_classes.Agent("vector", "Male", agents_personality_traits['vector'], system_prompt_vector, system_prompt_vector2, agent_config[4]["active"], analysis_model, vision_model2, analysis_context_length, agent_config[-1]['speaker_wav'], agent_config[4]["extraversion"])

# Task Agent, for agentic purposes, like Search Mode.
vectorAgent = agent_classes.VectorAgent(language_model, analysis_model, vision_model2, analysis_context_length, None, None)

### LIST OF AGENTS PRESENT IN THE CONVERSATION.
# REMEMBER, SOME OF THESE AGENTS MAY OR MAY NOT HAVE THINKING MODELS ASSIGNED AND MAY OR MAY NOT RESPOND DEPENDING ON ANALYSIS MODE STATUS.
agents = [
    axiom,
    axis,
    fractal,
    sigma,
    vector
    ]
