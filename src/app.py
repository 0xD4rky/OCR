import gradio as gr
import google.generativeai as genai
from PIL import Image
import pytesseract
import re

# Configure the generative AI model
genai.configure(api_key="")
model = genai.GenerativeModel('gemini-1.5-flash')

# Helper function to clean and structure the extracted text
def parse_label_text(text):
    # Use regex to extract specific sections from the text
    product_name = re.search(r"Product Name: (.+)", text)
    ingredients = re.search(r"Ingredients: (.+)", text)
    nutrition_info = re.search(r"Nutritional Information: (.+)", text)
    brand_name = re.search(r"Brand Name: (.+)", text)
    
    # Optional: Implement more parsing as needed for other label fields
    parsed_data = {
        "product_name": product_name.group(1) if product_name else "Unknown",
        "ingredients": ingredients.group(1) if ingredients else "Unknown",
        "nutrition_info": nutrition_info.group(1) if nutrition_info else "Unknown",
        "brand_name": brand_name.group(1) if brand_name else "Unknown"
    }
    
    return parsed_data

# Function to perform health analysis based on ingredients and nutritional data
def health_analysis(parsed_data):
    ingredients = parsed_data.get("ingredients", "")
    nutrition_info = parsed_data.get("nutrition_info", "")
    
    # Generate health analysis text
    prompt = f"""
    Analyze the following product based on its ingredients and nutritional information:
    
    Ingredients: {ingredients}
    Nutritional Information: {nutrition_info}
    
    Is this product healthy? Does it comply with common dietary restrictions such as diabetes, allergen-friendly, or low sugar? Provide a grade for this product.
    """
    
    response = model.generate_content([prompt])
    return response.text

# Function to handle image input, perform OCR, and generate text output
def gen_text(image):
    # Convert image to grayscale and perform OCR to extract text
    pil_image = Image.fromarray(image)
    extracted_text = pytesseract.image_to_string(pil_image)
    
    # Parse the extracted text
    parsed_data = parse_label_text(extracted_text)
    
    # Generate health analysis based on parsed data
    analysis_text = health_analysis(parsed_data)
    
    return analysis_text

# Define Gradio Interface
img = gr.Image()

intf = gr.Interface(fn=gen_text, inputs=img, outputs="text")

intf.launch(inline=False, share=True)
