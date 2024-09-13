from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import pytesseract
from io import BytesIO
import re

def get_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def perform_ocr(image):
    return pytesseract.image_to_string(image)

def answer_question(image, question, ocr_text):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # Prepare inputs
    encoding = processor(image, question, return_tensors="pt")

    # Forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    vqa_answer = model.config.id2label[idx]

    # If the question is about weight, search for weight information in OCR text
    if "weight" in question.lower():
        weight_pattern = r'\b(\d+(?:\.\d+)?)\s*(mg|g|kg)\b'
        weight_matches = re.findall(weight_pattern, ocr_text, re.IGNORECASE)
        if weight_matches:
            return f"{weight_matches[0][0]} {weight_matches[0][1]}"

    return vqa_answer  # Return VQA answer if no better match found

# Example usage
url = r'https://m.media-amazon.com/images/I/71jBLhmTNlL.jpg'
question = "What is the item weight?"

image = get_image_from_url(url)
ocr_text = perform_ocr(image)
answer = answer_question(image, question, ocr_text)

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"OCR Text: {ocr_text}")