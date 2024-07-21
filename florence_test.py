import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
model.to(device)  # Move model to GPU if available

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

prompt = "<OCR>"

# Load the image
picture = "axiom_screenshot.png"
image = Image.open(picture)

# Preprocess the inputs
inputs = processor(text=prompt, images=image, return_tensors="pt")

print("Viewing image...")
# Generate the output
generated_ids = model.generate(
    input_ids=inputs["input_ids"].cuda(),
    pixel_values=inputs["pixel_values"].cuda(),
    max_new_tokens=1024,
    do_sample=True,
    temperature=1,
    num_beams=10
)

print("Processing...")
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(
        generated_text, 
        task='<OCR>', 
        image_size=(image.width, image.height)
    )

print(parsed_answer)
