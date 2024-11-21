# Vector Companion

Your friendly AI Companion, with two distinct personalities: Axiom and Axis! Here to accompany you everywhere you go on your computer!

![image](https://github.com/user-attachments/assets/11cbbdec-51fb-4551-938a-3ff40fe4432f)

![image](https://github.com/user-attachments/assets/f14a50e5-74e4-48a9-8e82-d9c0b5432b2a)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Demo](https://www.youtube.com/watch?v=V8dWY1K61-0)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting VB Cable and Microphone Issues](#troubleshooting-vb-cable-and-microphone-issues)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Whether playing games, watching videos, or browsing online, Axiom and Axis will talk to each other about whatever you're doing and will also talk to you directly! The motivation behind this project is to create not one, but two multimodal virtual companions who are very lifelike, responsive, and charming. They can see, hear, and speak about anything presented to them on screen!

They transcribe audio output and user microphone input simultaneously while periodically taking screenshots and viewing/reading OCR text on screen. They use this information to form a conversation, with the ability to remember past key events and summarize the conversation history periodically, enabling them to pick up where you left off.

## Features

- Can view images periodically, captioning images and reading text via OCR.
- Can hear and transcribe computer audio in real-time (English only due to base model size Whisper language limitations but you may replace base with WhisperV3 Turbo for multilingual support).
- Can hear user microphone input in real-time (English only due to base model size Whisper language limitations but you may replace base with WhisperV3 Turbo for multilingual support).
- Voice Cloning enables distinct voice output generation for agents Axiom and Axis.

## Installation

### Prerequisites

**Note:** This framework is designed to run on Windows only.

- Minimum 20GB VRAM required to run the entire framework on `gemma2:2b-instruct-q8_0`. You may swap out the model in the `language_model` variable in `main.py` if you would like to lower the VRAM usage or replace the model with a larger one.
- You will need a `torch` version compatible with your CUDA version installed.
- You need to install Ollama if you haven't already and download `gemma2:2b-instruct-q8_0` or whichever model you'd like to use in Ollama. You must also install `minicpm-v:8b-2.6-q4_0` (or greater size) in order to use updated vision component.
- You need VB Cable installed on your PC in order for Python to listen to your own PC.

### Cloning

```
git clone https://github.com/SingularityMan/vector_companion.git
cd vector_companion
conda create --name vector_companion
conda activate vector_companion
pip install -r requirements.txt
```
### Configuring VB Cable-Compatible Headset
On Windows, install VB Cable so Python can listen to the computer's audio output in real-time.
Once installed, link VB Cable to the device (presumably your headset) via Windows Sound settings so Python can accurately capture the sound.
Ensure VB Cable is selected as your speaker once it is configured, otherwise Vector Companion won't be able to hear audio.

### Usage
After meeting the necessary prerequisites, installing the required dependencies, and configuring then troubleshooting any VB Cable issues (listed below), simply run main.py.

```
conda activate vector_companion
python main.py
```
### Troubleshooting VB Cable and Microphone Issues
If you successfully installed VB Cable and correctly linked it to the device of your choice but the framework isn't capturing computer audio (and therefore not transcribing it), review the `input_device_index` keyword parameter in `p.open()` under the `record_audio_output()` function in `config/config.py`. The script may or may not link to the correct device index. If you have multiple devices, it might incorrectly pick the wrong one. If this is the case, try manually assigning the index starting from index 0 and work your way up. You may receive streaming errors with each attempt:

```
def record_audio_output(audio, WAVE_OUTPUT_FILENAME, FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, file_index_count):

    global can_speak
    file_index = 0

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
                        input_device_index=device_index) <------------------------------------- device_index

        print("* recording Audio Transcript")
```

You will also most likely need to modify the `input_device_index` keyword argument under `audio.open()` in `record_audio()` to select the right microphone index for your headset, which is set to 1 by default. If no audio input is being captured, follow similar steps with `record_audio()` in `config/config.py`:

```
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
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=1, frames_per_buffer=CHUNK) <------------------------------- device index
            #print("waiting for speech...")
            frames = []
            image_path = None
```
# Installing flash_attn on Windows
### Installing Microsoft Visual Studio Code 
Install Microsoft Visual Studio Code's latest version (2022) that is compatible with your CUDA version (12.2 or greater) and in the installer, make sure to include these capabilities:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools (x86 & x84)
   - C++ CMake Tools for Windows
   - Windows 10 or Windows 11 SDK, depending on your Windows OS

### Installing Torch/Cuda
After that, install a version of `torch` compatible with your CUDA version that is compatible with MSVC.

### Building flash_attn from source
Lastly, make sure to carefully edit then run `flash-attn-clone-source-compile-stable-torch.bat` in order to build flash_attn from source. Make sure to edit the file to compile flash_attn for a version compatible with your torch/CUDA and your python version.

### Contributing
Contributions are welcome! You may follow our [contribution](CONTRIBUTING.md) guides here.

### License
This project is licensed under the [Vector Companion License](LICENSE.md) - see the LICENSE file for details.

Note: This project includes the XTTSv2 model, which is licensed under a restrictive license. Please refer to the XTTSv2 License for specific terms and conditions.
