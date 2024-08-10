import base64
import PyPDF2
import pytesseract
import pandas as pd
from PIL import Image
from io import BytesIO

def get_pdf_data(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    
    data = f"PDF file name: {pdf_path.split('/')[-1].replace('.pdf', '')}\nContent: {text}"
    return data

def get_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    text = df.head().to_string()
    data = f"Excel file name: {excel_path.split('/')[-1].replace('.xlsx', '')}\nContent: {text}"
    return data

def get_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    text = df.head().to_string()
    data = f"CSV file name: {csv_path.split('/')[-1].replace('.csv', '')}\nContent: {text}"
    return data

def get_txt_data(txt_path):
    with open(txt_path, 'r') as file:
        text = file.read()

    data = f"Text file name: {txt_path.split('/')[-1].replace('.txt', '')}\nContent: {text}"
    return data

def get_image_data(image_path):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
        
    image = image.resize((512, 512))

    extracted_text = pytesseract.image_to_string(image)
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    data_dict = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}",
            }
        }
    
    return extracted_text, data_dict