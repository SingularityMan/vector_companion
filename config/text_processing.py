from __future__ import annotations
from collections import Counter
import re
import asyncio

import numpy as np
import noisereduce as nr
import soundfile as sf

def split_buffer_into_sentences(buffer):
    """
    Splits buffer into sentences and returns the remaining buffer.
    """
    sentence_endings = re.compile(r'([.!?>])')
    sentences = []
    while True:
        match_string = sentence_endings.search(buffer)
        if match_string:
            end = match_string.end()
            sentence = buffer[:end].strip()
            sentences.append(sentence)
            buffer = buffer[end:]
        else:
            break
    return sentences, buffer

def clean_text(text):
    """
    Cleans the text by removing unwanted characters and patterns.
    """
    text = re.sub(r'"', '', text)
    #text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\((.*?)\)', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = text.replace('\\n', '')
    return text.strip()

async def synthesize_sentence(tts, sentence, speaker_wav, sample_rate):
    """
    Asynchronously synthesizes a sentence and returns the audio data.
    """
    try:
        # Preprocess the sentence to remove problematic characters
        sentence = sentence.strip().strip("'\"")
        sentence = re.sub(r'[\'\"]', '', sentence)

        loop = asyncio.get_event_loop()
        # Synthesize the sentence using GPU
        audio = await loop.run_in_executor(
            None, lambda: tts.tts(text=sentence, speaker_wav=speaker_wav, language="en")
        )
        # Check the type of 'audio' and process accordingly
        if isinstance(audio, list):
            # Convert list to NumPy array
            audio_array = np.array(audio, dtype=np.float32)
        elif isinstance(audio, np.ndarray):
            # If it's already a NumPy array
            audio_array = audio.astype(np.float32)
        else:
            raise TypeError(f"Unexpected audio type: {type(audio)}")

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
        y=audio_array,
        sr=sample_rate,
        prop_decrease=0.0,
        freq_mask_smooth_hz=1000,
        time_mask_smooth_ms=100,
        stationary=True,
        n_fft=512,
        win_length=512,
        use_torch=True,
        device='cuda'
    )

        return reduced_noise
        #return audio
    except Exception as e:
        print(f"Error during TTS synthesis for sentence '{sentence}': {e}")
        return None

def find_repeated_words(text: str, threshold: int = 6) -> str:
    pattern = r'\b(\w+)\b'
    words = re.findall(pattern, text, re.IGNORECASE)
    word_counts = Counter(words)
    filtered_words = [word for word in words if word_counts[word.lower()] < threshold]
    word_counts.clear()
    return " ".join(filtered_words)

def remove_repetitive_phrases(text: str) -> str:
    pattern = re.compile(r'(\b\w+\b(?:\s+\b\w+\b){0,4})(?:\s+\1)+', re.IGNORECASE)
    result = pattern.sub(r'\1', text)
    return result

def check_sentence_length(text: str, sentence_length: int = 2) -> Union[Tuple[List[str], str], str]:
    text = remove_repetitive_phrases(text)
    sentences = re.split(r'(?:(?<=[.?;!]))\s', text.strip())
    if sentences:
        if len(sentences) > 1:
            return (sentences[:sentence_length], ' '.join(sentences[:sentence_length]))
        else:
            return sentences, sentences[0]
    return "No valid text found."
