import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load the pre-trained Longformer model and tokenizer
model_name = 'valhalla/longformer-base-4096-finetuned-squadv1'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create the QA pipeline
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, device=device)

# Load your dataset
df = pd.read_csv(r'C:\Users\DELL\Amazon\unextracted.csv')

# Prepare the data
df['context'] = df['text'].fillna('').astype(str)
df['question'] = df['entity_name'].fillna('').astype(str)

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'["“”]', '', text)
    text = re.sub(r'[\u201c\u201d\u201e\u201f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['cleaned_context'] = df['context'].apply(preprocess_text)

# Refine the question
def refine_question(entity_name):
    question_map = {
        'width': 'What is the width?',
        'height': 'What is the height?',
        'depth': 'What is the depth?',
        'item_weight': 'What is the item weight?',
        'maximum_weight_recommendation': 'What is the maximum weight recommendation?',
        'voltage': 'What is the voltage?',
        'wattage': 'What is the wattage?',
        'item_volume': 'What is the item volume?',
        # Add more mappings as needed
    }
    return question_map.get(entity_name.lower(), f'What is the {entity_name}?')

df['refined_question'] = df['question'].apply(refine_question)

# Define the extraction function
def extract_entity_value(row):
    context = row['cleaned_context']
    question = row['refined_question']
    if not context or not question:
        return ''
    try:
        # Limit context length to model's max input size
        max_length = tokenizer.model_max_length
        inputs = tokenizer.encode_plus(question, context, max_length=max_length, truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            # Get the most likely beginning and end of answer
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1  # Add 1 since slicing is exclusive

            # Convert tokens to answer
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
            answer = answer.strip()

            # Filter out invalid answers
            if answer.lower() in ['no', 'none', 'n/a', 'not applicable', 'unavailable']:
                return ''
            return answer
    except Exception as e:
        return ''

# Process the data in batches
batch_size = 1000  # Adjust based on your system
num_rows = len(df)
results = []

for start_idx in range(0, num_rows, batch_size):
    end_idx = min(start_idx + batch_size, num_rows)
    batch_df = df.iloc[start_idx:end_idx].copy()
    print(f'Processing rows {start_idx} to {end_idx}')
    batch_df['extracted_value'] = batch_df.apply(extract_entity_value, axis=1)
    results.append(batch_df)

# Combine results
df_result = pd.concat(results, ignore_index=True)

# Save the results
df_result.to_csv('extracted_entity_values.csv', index=False)
print('Extraction complete. Results saved to extracted_entity_values.csv')
