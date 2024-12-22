# Instruction Manual

Vector Companion is a very powerful and versatile but may be hard to understand how to use. This guide will provide operating instructions.

## Speaking Modes

In the current version, Vector Companion is divided between two different modes: `Chat Mode` and `Analysis Mode.`

To enable and disable analysis mode, simply including "Analysis mode on" or "Analysis Mode off" in your message. Depending on your available VRAM, the model you choose and Ollama's configuration, Ollama will either load the model or replace the `language_model` with the `analysis_model` in `main.py` and vice versa.

### Chat Mode

Chat mode activates all the agents in the `agents` list in `main.py` that contain the `language_model`. All other agents without it will not speak. When Chat Mode is enabled:

- Chat Mode will wait a certain amount of time in random intervals before speaking or until the user speaks. You can modify this by adjusting the `file_index_count` variable and the `random_record_seconds` variable under `main()` in `main.py`.
- If the user speaks, the user will be allowed to speak for as long as he'd like, but due to `Whisper's` limitations, only 30 seconds of user input will be transcribed.
- If the user doesn't speak after the recording has been completed, the agents will speak to each other instead about the current situation.
- Each time the user speaks, there will be a one-sentence entry about the user's personality stored under the `user_memory.json` file. Due to context limitations, this has been set to a default of 5 entries allowed at any given time. You can modify this in `process_user_memory` in `main.py`.
- You can also command only one agent to speak by mentioning their name directly, at which point only any agents named in the user's message will speak.

### Analysis Mode

Analysis mode activates all the agents in the `agents` list in `main.py` that contain the `analysis_model`. All other agents will not speak. When Analysis Mode is enabled:

- All agents containing the `analysis_model` (recommend restricting to 1 agent with this capability) will not speak to the user unless spoken to. 
- Once the user speaks, the analysis model (default `qwq:32b-preview-q8_0` but you can replace this by changing the model under `analysis_model`) will analyze the user's message along with the contextual information (screenshots, audio, etc.) using COT reasoning process, if applicable. This will cause the analysis_model to be muted while it processes its thoughts until it reaches a Final Solution and speaks to the user. The user's personality traits will not be updated nor introduced to the analysis model. 

## Modifying Agent traits

This framework is modular, allowing the developer to modify it as they see fit. Here are the different components available for change in `main.py`:

- `language_model` - used for Chat Mode.
- `analysis_model` - used for Analysis Mode.
- `agent_voice_samples` - This directory contains the voice samples for the agents. You can add or remove as many as you'd like. Ensure to include the correct name belonging to each agent in `agent_config` in `main.py`.
- `agents` - Contains all the agents in your framework.
- `config.Agent` is a class that defines your agent. See `Agent` class in `config/config.py` for more details on their attributes.
- `agent_config` - Define your agents in this list of dictionaries. Extraversion defines how often they speak in Chat Mode, which is defined on a scale between 0 and 1.
- `agents_personality_traits` - This defines the agents' personality traits in Chat Mode. You can add or remove as many categories as you'd like but their traits will be shuffled for each response.

- `model_name` - This is reserved for whichever Whisper model you'd like to use (tiny, base, small, medium, large, turbo).

## Configuring Ollama

Ollama recently updated its framework to introduce a number of improvements. I highly recommend updating the `system environment variables` to introduce the following:

- `OLLAMA_FLASH_ATTENTION=1` - Enables `flash_attn`, speeding up inference.
- `OLLAMA_KV_CACHE_TYPE=Q8_0` - Significantly reduces required VRAM with little loss in performance. q4_0 is buggy and introduces noticeable loss but will be updated at a later date.
- `OLLAMA_MAX_LOADED_MODELS=2` - To allow for the vision model and whichever language model to be active simultaneously. You can increase or decrease depending on your needs.

