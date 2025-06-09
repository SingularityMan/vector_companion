from __future__ import annotations
import time

import pyautogui as pygi
import asyncio
import ollama
from config.text_processing import clean_text, split_buffer_into_sentences

image_lock = False

def write_sentence(sentence: str):
    with open("screenshot_description.txt", "a", encoding="utf-8") as f:
        f.write("\n" + sentence)

# Vision model view images:
async def view_image(vision_model: str, vision_context_length: int, encoded_image: Any, can_speak_event: Any):
    global image_lock
    try:
        prompt = "Describe the contents on the screen."
        loop = asyncio.get_event_loop()
        start_time = time.time()

        def run_generate():
            # Call ollama.generate with stream enabled.
            return ollama.generate(
                model=vision_model,
                prompt=prompt,
                images=[encoded_image],
                options={
                    "repeat_penalty": 1.15,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": vision_context_length,
                    "num_batch": 512,
                    "num_predict": 500
                },
                stream=True  # ensure streaming is enabled
            )

        # Offload the blocking generate call to the executor
        stream = await loop.run_in_executor(None, run_generate)

        buffer = ""
        complete_response = ""
        # Iterate over the streamed chunks
        for chunk in stream:
            if time.time() - start_time > 60:
                image_lock = False
                break
            # Yield control to let other tasks run
            await asyncio.sleep(0.10)
            # Each chunk is assumed to be a dict with a key (e.g., "response")
            content = chunk.get("response", "")
            if content:
                buffer += content
                # Process the buffer into sentences (or any other chunking logic)
                sentences, buffer = split_buffer_into_sentences(buffer)
                for sentence in sentences:
                    sentence = clean_text(sentence)
                    # Offload file writing so it doesn't block the event loop
                    await loop.run_in_executor(None, write_sentence, sentence)
                    complete_response += sentence + " "
                    yield sentence  # yield each sentence as it is ready

        # Process any remaining buffer
        if buffer.strip():
            buffer = clean_text(buffer)
            complete_response += buffer + " "
            yield buffer

    except Exception as e:
        image_lock = False
        print("Error:", e)

# Example of how you might run this generator and handle cancellation:
async def run_image_description(vision_model, vision_context_length, can_speak_event):
    global image_lock
    print("[SCREENSHOT TAKEN]")
    image_picture = pygi.screenshot("axiom_screenshot.png")
    with open("axiom_screenshot.png", "rb") as image_file:
        encoded_image = image_file.read()
    gen = view_image(vision_model, vision_context_length, encoded_image, can_speak_event)
    try:
        async for sentence in gen:
            with open("screenshot_description.txt", "r", encoding="utf-8") as f:
                image_text = f.read()
                if len(image_text.split()) >= vision_context_length:
                    image_text = " ".join(image_text.split()[len(image_text.split())//2:])
                    with open("screenshot_description.txt", "w", encoding="utf-8") as f:
                        f.write(image_text)
            if not can_speak_event.is_set():
                image_lock = False
                break
    except asyncio.CancelledError:
        pass
    finally:
        image_lock = False
