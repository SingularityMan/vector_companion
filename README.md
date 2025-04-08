# Vector Companion

Your friendly AI Companions, Here to accompany you everywhere you go on your computer!

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

Whether playing games, watching videos, or browsing online, These agents will talk to you and each other about whatever you're doing and will also talk to you directly when prompted! They can also perform online searches and even deep searches! The motivation behind this project is to create not one, but multiple multimodal virtual companions who are very lifelike, responsive, and charming. They can see, hear, and speak about anything presented to them on screen!

They transcribe audio output and user microphone input simultaneously while periodically taking screenshots and viewing/reading OCR text on screen. They use this information to form a conversation and chat about anything and everything under the sun!

## Features

- Can view images periodically, captioning images and reading text via OCR.
- Can hear and transcribe computer audio in real-time (English only due to base model size Whisper language limitations but you may replace base with WhisperV3 Turbo for multilingual support).
- Can hear user microphone input in real-time (English only due to base model size Whisper language limitations but you may replace base with WhisperV3 Turbo for multilingual support).
- Voice Cloning enables distinct voice output generation for the agents.
- Have web search functionality enabled via duckduckgo_search.

## Installation

### Prerequisites

- VRAM requirements vary greatly, and it all depends on which models you use. To see which models are being used, please see `config.py`. You can swap these models out at any time.
- You will need a `torch` version compatible with your CUDA version installed (for Mac just install torch).
- You need to install Ollama if you haven't already and download any models you'd like to use in Ollama.

## Audio Loop back

### Windows:

  - You will need VB-Cable installed on your PC to handle the audio loopback. 
  - Once installed, link it to your headset via sound settings.

### MacOS

  - Same as Windows, but instead you can install `Blackhole` or `Soundflower`

### Linux

  - You will need to create a Virtual Sink (Null Sink) to enable audio loop back. Simply run this command:

   `pactl load-module module-null-sink sink_name=VirtualLoopback sink_properties=device.description=VirtualLoopback`
  
  - Then you may set it up as the default sink via the following:

   `pactl set-default-sink VirtualLoopback`

  - Alternatively, you can use `pavucontrol` to route specific application outputs to the VirtualLoopback device.

  - Finally, If you want this setup available each time you start your system, you can add the module load command to your `PulseAudio` configuration. 
  Editing your `default.pa` (usually found in `/etc/pulse/` or `~/.config/pulse/`) and adding the following:
   
  `load-module module-null-sink sink_name=VirtualLoopback sink_properties=device.description=VirtualLoopback`

## Cloning

```
git clone https://github.com/SingularityMan/vector_companion.git
cd vector_companion
conda create --name vector_companion
conda activate vector_companion
pip install -r requirements.txt
```

### Usage
After meeting the necessary prerequisites, installing the required dependencies, and configuring then troubleshooting any VB Cable issues (listed below), simply run `activate.bat` or `main.py`.

```
conda activate vector_companion
python main.py
```

# Installing Flash Attention

## Linux

`pip install flash-attn`

## Windows

Windows does not have official support for Flash Attention, but it is impossible to install it.
This will be a very lengthy and difficult process, so following each step carefully will get you there.

### Installing Microsoft Visual Studio Code 
Install Microsoft Visual Studio Code's latest version (2022) that is compatible with your CUDA version (12.2 or greater) and in the installer, make sure to include these capabilities:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools (x86 & x84)
   - C++ CMake Tools for Windows
   - Windows 10 or Windows 11 SDK, depending on your Windows OS

### Installing Torch/Cuda
After that, install a version of `torch` compatible with your CUDA version that is compatible with MSVC.

### Building flash_attn from source
Lastly, make sure to carefully edit then run `flash-attn-clone-source-compile-stable-torch.bat` in order to build flash_attn from source. Make sure to edit the file to compile flash_attn for a version compatible with your torch/CUDA and python version.

### Contributing
Contributions are welcome! You may follow our [contribution](CONTRIBUTING.md) guides here.

### License
This project is licensed under the [Vector Companion License](LICENSE.md) - see the LICENSE file for details.

Note: This project includes the XTTSv2 model, which is licensed under a restrictive license. Please refer to the XTTSv2 License for specific terms and conditions.
