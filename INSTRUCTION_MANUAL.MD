# Instruction Manual

Vector Companion is a very powerful and versatile but may be hard to understand how to use or configure. This guide will provide operating instructions.

## Speaking Modes

In the current version, Vector Companion is divided between two different modes: `Chat Mode` and `Analysis Mode.`

To enable and disable analysis mode, simply including "Analysis mode on" or "Analysis Mode off" in your message. Depending on your available VRAM, the model you choose and Ollama's configuration, Ollama will either load the model or replace the `language_model` with the `analysis_model` in `config.py` and vice versa.

### Chat Mode

Chat mode activates all the agents in the `agents` list in `main.py` that contain the `language_model`. All other agents without it will not speak. When Chat Mode is enabled:

- Chat Mode will listen to user voice input or PC audio output. Once it detects intelligible audio, it will keep listening until no more audio is heard, at which point it will generate a response.
- If the user speaks, the user will be allowed to speak for as long as he likes.
- If the user doesn't speak after the recording has been completed, the agents will speak to each other instead about the current situation or by themselves if only one agent is active.
- You can also command only one agent to speak by mentioning their name directly, at which point only the agents named in the user's message will speak.

If you want to mute the agents in Chat Mode, just say "Mute Mode On" and they will not speak unless spoken to.

### Analysis Mode

Analysis mode activates all the agents in the `agents` list in `config.py` that contain the `analysis_model`. All other agents will not speak. When Analysis Mode is enabled:

- All agents containing the `analysis_model` (recommend restricting to 1 agent with this capability) will not speak to the user unless spoken to. 
- Once the user speaks, the analysis model will analyze the user's message along with the contextual information (screenshots, audio, etc.) using COT reasoning process, if applicable. This will cause the analysis_model to be muted while it processes its thoughts until it reaches an answer and speaks to the user. The user's personality traits will not be updated nor introduced to the analysis model. 

### Adding and Removing agents.

You can add and remove agents from the chat via voice commands at any time by saying `add <agent name>` and `remove <agent name>`. 
This must be done for each agent you wish to add and remove.

### Search Mode

You can toggle Search Mode by saying "Search Mode On" and "Search Mode Off". 
When Search Mode is enabled, `vectorAgent` will determine based on the user's message and conversation history if a web search is required.
`vectorAgent` will perform one of the following actions:

- `Simple Search` - This performs a simple query to `duckduckgo_search` and returns the text results from each result from `duckduckgo_search`.
- `Deep Search` - The agent will recursively extract text from each search result and explore different links inside each search result until it no longer finds relevant links or is unable to proceed.
It will then use praw to perform a search on reddit for relevant posts and threads to answer the user's inquiry.
- `No Search` - The agent determined an online search is unecessary and will continue the conversation.

Note: If multiple agents are in the chat and search mode is enabled, the search will be performed only once, regardless of category. 
vectorAgent will still determine if a search is necessary or not, but it will evaluate that decision each time it is an agent's turn to speak so long as the search hasn't been attempted yet.

### Chaining Commands together.

You can actually include multiple commands in one voice message by simply speaking each command in the correct format in the same message. 
For Example:

```
"Analysis mode off, add Axiom, add Axis, remove Sigma, search mode on, perform a deep search, etc."
```
This example command will disable Analysis Mode, add Axiom and Axis to the chat, remove Sigma from the chat, enable Search Mode and perform a deep search.


## Modifying Agent traits

This framework is modular, allowing the developer to modify it as they see fit. Here are the different components available for change in `config.py`:

- `language_model` - used for Chat Mode.
- `analysis_model` - used for Analysis Mode.
- `vision_model1` and `vision_model2` which are the vision components for Chat Mode and Analysis Mode.
- `agent_voice_samples` - This is a directory that contains the voice samples for the agents. You can add or remove as many as you'd like. Ensure to include the correct name belonging to each agent in `agent_config` in `config.py`.
- `agents` - Contains all the agents in your framework.
- `Agent` is a class that defines your agent. See `Agent` class in `config/agent_classes.py` for more details on their attributes.
- `agent_config` - Define your agents in this list of dictionaries. Extraversion defines how often they speak in Chat Mode, which is defined on a scale between 0 and 1.
- `agents_personality_traits` - This defines the agents' personality traits. You can add or remove as many categories as you'd like but their traits will be shuffled each time a response is generated.
- `audio_model_name` - This is reserved for whichever Whisper model you'd like to use (tiny, base, small, medium, large, turbo).

## Configuring Ollama

Ollama recently updated its framework to introduce a number of improvements. I highly recommend updating the `system environment variables` to introduce the following:

- `OLLAMA_FLASH_ATTENTION=1` - Enables `flash_attn`, speeding up inference.
- `OLLAMA_KV_CACHE_TYPE=Q8_0` - Significantly reduces required VRAM with little loss in performance. q4_0 is buggy and introduces noticeable loss but will be updated at a later date.
- `OLLAMA_MAX_LOADED_MODELS=2` - To allow for the vision model and whichever language model to be active simultaneously. You can increase or decrease depending on your needs.
- `OLLAMA_MAX_QUEUE_MODELS=3` - Determines how many generations can occur simultaenously.

