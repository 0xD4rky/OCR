import pandas as pd
import numpy as np
import requests
from PIL import Image
import pytesseract
import cv2
from io import BytesIO
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from tqdm import tqdm

# Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust this line if Tesseract is not in your PATH

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def localize_text(image):
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find contours in the thresholded image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Filter contours based on area to remove noise
    min_area = 100
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
    
    return boxes

def recognize_text(image, boxes):
    text_results = []
    for box in boxes:
        x, y, w, h = box
        roi = image.crop((x, y, x + w, y + h))
        text = pytesseract.image_to_string(roi, config='--psm 6')
        text_results.append(text.strip())
    return ' '.join(text_results)

def validate_text(text, regex_pattern):
    match = re.search(regex_pattern, text)
    return match.group() if match else None

def process_row(row, regex_pattern):
    image_url = row['image_link']
    image = download_image(image_url)
    
    if image is None:
        return None
    
    text_boxes = localize_text(image)
    recognized_text = recognize_text(image, text_boxes)
    validated_text = validate_text(recognized_text, regex_pattern)
    
    return validated_text

def process_csv(csv_path, regex_pattern, max_workers=10):
    df = pd.read_csv(csv_path)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_row, row, regex_pattern): idx 
                         for idx, row in df.iterrows()}
        
        for future in tqdm(as_completed(future_to_row), total=len(df), desc="Processing rows"):
            idx = future_to_row[future]
            try:
                result = future.result()
                results.append({'index': idx, 'validated_text': result})
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                results.append({'index': idx, 'validated_text': None})
    
    return pd.DataFrame(results).sort_values('index')

def prepare_data(df):
    X = df['validated_text'].fillna('')
    y = df['entity_name']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, X_val, y_val, max_iter=100):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=max_iter)
    
    best_f1 = 0
    best_model = None
    
    for epoch in range(max_iter):
        model.fit(X_train_vec, y_train)
        
        # Compute loss
        train_loss = log_loss(y_train, model.predict_proba(X_train_vec))
        val_loss = log_loss(y_val, model.predict_proba(X_val_vec))
        
        # Compute F1 score
        val_pred = model.predict(X_val_vec)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        print(f"Epoch {epoch+1}/{max_iter}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
        
        # Early stopping condition (you can adjust this)
        if epoch > 10 and val_f1 < best_f1 * 0.95:
            print("Early stopping")
            break
    
    return best_model, vectorizer

# Main execution
if __name__ == "__main__":
    csv_path = r'C:\Users\DELL\Amazon\5000rows.csv'
    regex_pattern = r'\b(\d+(?:\.\d+)?)\s*(centimetre|foot|inch|metre|millimetre|yard|gram|kilogram|microgram|milligram|ounce|pound|ton|kilovolt|millivolt|volt|kilowatt|watt|centilitre|cubic foot|cubic inch|cup|decilitre|fluid ounce|gallon|imperial gallon|litre|microlitre|millilitre|pint|quart)s?\b'
    
    # Process CSV and extract text
    results_df = process_csv(csv_path, regex_pattern)
    
    # Merge results with original DataFrame
    original_df = pd.read_csv(csv_path)
    final_df = pd.merge(original_df, results_df, left_index=True, right_on='index')
    
    # Prepare data for training
    X_train, X_val, y_train, y_val = prepare_data(final_df)
    
    # Train model
    best_model, vectorizer = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    X_val_vec = vectorizer.transform(X_val)
    val_pred = best_model.predict(X_val_vec)
    final_f1 = f1_score(y_val, val_pred, average='weighted')
    print(f"Final Validation F1 Score: {final_f1:.4f}")
    
    # Save results and model
    final_df.to_csv('processed_results.csv', index=False)
    print("Processing complete. Results saved to 'processed_results.csv'")
    
    # You might want to save the model and vectorizer for later use
    # import joblib
    # joblib.dump(best_model, 'text_classification_model.joblib')
    # joblib.dump(vectorizer, 'text_vectorizer.joblib')