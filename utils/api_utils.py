import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from utils.prompts import generation_prompt
from utils.parser import (
    get_image_data, 
    get_pdf_data, 
    get_excel_data, 
    get_csv_data, 
    get_txt_data
)

api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def generate_description(file, message=''):
    content = [{"role": "system", "content": generation_prompt}]
    text = ""

    if file.endswith('.pdf'):
        text = get_pdf_data(file)
        content.append({"role": "user", "content": text})
    elif file.endswith('.txt'):
        text = get_txt_data(file)
        content.append({"role": "user", "content": text})
    elif file.endswith('.csv'):
        text = get_csv_data(file)
        content.append({"role": "user", "content": text})
    elif file.endswith('.xlsx'):
        text = get_excel_data(file)
        content.append({"role": "user", "content": text})
    elif file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.JPG')):
        extracted_text, img_data = get_image_data(file)
        text_data = {"type": "text", "text": extracted_text}
        content.append({"role": "user", "content": [img_data, text_data]})

    if message:
        content.append({"role": "user", "content": message})

    response = run_api(content)
    
    return response, text

def run_api(messages):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        max_tokens=500,
    )
    return response.choices[0].message.content