import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytesseract
from PIL import Image
import requests
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import cv2
import re
import time
from io import BytesIO
import torch.cuda as cuda

# Define the regex pattern
pattern = r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)?\s*((?:kilo|centi|milli|micro)?(?:gram|metre|litre|volt|watt|ounce|pound|ton|foot|inch|yard|gallon|pint|quart|cup)s?|fl\.?\s*oz\.?|cubic (?:foot|inch))'

def extract_measurements(text):
    matches = re.findall(pattern, text, re.IGNORECASE)
    results = []
    for match in matches:
        start_value = float(match[0])
        end_value = float(match[1]) if match[1] else None
        unit = match[2].lower().replace('.', '').replace('fl oz', 'fluid ounce')
        
        # Normalize units
        if unit in ['g', 'gram', 'grams']:
            unit = 'gram'
        elif unit in ['kg', 'kilogram', 'kilograms']:
            unit = 'kilogram'
        elif unit in ['mg', 'milligram', 'milligrams']:
            unit = 'milligram'
        elif unit in ['µg', 'mcg', 'microgram', 'micrograms']:
            unit = 'microgram'
        elif unit in ['l', 'liter', 'litre', 'litres']:
            unit = 'litre'
        elif unit in ['ml', 'milliliter', 'millilitre']:
            unit = 'millilitre'
        elif unit in ['oz', 'ounce', 'ounces']:
            unit = 'ounce'
        elif unit in ['lb', 'lbs', 'pound', 'pounds']:
            unit = 'pound'
        elif unit in ['v', 'volt', 'volts']:
            unit = 'volt'
        elif unit in ['w', 'watt', 'watts']:
            unit = 'watt'
        
        if end_value:
            results.append((start_value, end_value, unit))
        else:
            results.append((start_value, unit))
    
    return results

class TextDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_col='entity_value'):
        self.dataframe = dataframe
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_text = extract_text_from_image(row['image_link'])
        label = row[self.label_col]
        if self.transform:
            image_text = self.transform(image_text)
        sample = {'text': image_text, 'label': label}
        return sample

def extract_text_from_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('L')  # convert to grayscale
    ocr_result = pytesseract.image_to_string(img)
    return ocr_result