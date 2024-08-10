import os
import base64
import PyPDF2
import pytesseract
import pandas as pd
from PIL import Image
from io import BytesIO
from typing import Tuple, Dict

def get_pdf_data(pdf_path: str) -> str:
    """
    Retrieves the content of a PDF file.

    Parameters:
    -----------
    pdf_path : str
        The path to the PDF file.

    Returns:
    --------
    str
        A string containing the name of the PDF file and its extracted text content.
    
    Raises:
    -------
    FileNotFoundError:
        If the specified PDF file does not exist.
    PyPDF2.errors.PdfReadError:
        If there is an error reading the PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""

    data = f"PDF file name: {os.path.basename(pdf_path).replace('.pdf', '')}\nContent: {text}"
    return data

def get_excel_data(excel_path: str) -> str:
    """
    Retrieves the content of an Excel file.

    Parameters:
    -----------
    excel_path : str
        The path to the Excel file.

    Returns:
    --------
    str
        A string containing the name of the Excel file and its content.
    
    Raises:
    -------
    FileNotFoundError:
        If the specified Excel file does not exist.
    ValueError:
        If the Excel file cannot be read.
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"The file {excel_path} does not exist.")
    
    try:
        df = pd.read_excel(excel_path)
    except ValueError as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")
    
    text = df.head().to_string()
    data = f"Excel file name: {os.path.basename(excel_path).replace('.xlsx', '')}\nContent: {text}"
    return data

def get_csv_data(csv_path: str) -> str:
    """
    Retrieves the content of a CSV file.

    Parameters:
    -----------
    csv_path : str
        The path to the CSV file.

    Returns:
    --------
    str
        A string containing the name of the CSV file and its content.
    
    Raises:
    -------
    FileNotFoundError:
        If the specified CSV file does not exist.
    pd.errors.ParserError:
        If the CSV file cannot be parsed.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error reading CSV file: {str(e)}")
    
    text = df.head().to_string()
    data = f"CSV file name: {os.path.basename(csv_path).replace('.csv', '')}\nContent: {text}"
    return data

def get_txt_data(txt_path: str) -> str:
    """
    Retrieves the content of a text file.

    Parameters:
    -----------
    txt_path : str
        The path to the text file.

    Returns:
    --------
    str
        A string containing the file name and the content of the text file.
    
    Raises:
    -------
    FileNotFoundError:
        If the specified text file does not exist.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"The file {txt_path} does not exist.")
    
    with open(txt_path, 'r') as file:
        text = file.read()

    data = f"Text file name: {os.path.basename(txt_path).replace('.txt', '')}\nContent: {text}"
    return data

def get_image_data(image_path: str) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """
    Processes an image by resizing it, extracting text using OCR, and converting it to a base64-encoded string.

    Parameters:
    -----------
    image_path : str
        The file path to the image.

    Returns:
    --------
    Tuple[str, Dict[str, Dict[str, str]]]
        A tuple containing:
        - extracted_text: str
            The text extracted from the image using OCR.
        - data_dict: Dict[str, Dict[str, str]]
            A dictionary with image data, including the base64-encoded image string.

    Raises:
    -------
    FileNotFoundError:
        If the specified image file does not exist.
    OSError:
        If the image file cannot be opened or processed.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    try:
        image = Image.open(image_path)
    except OSError as e:
        raise OSError(f"Error opening image file: {str(e)}")

    # Convert RGBA images to RGB to ensure consistency
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize((512, 512))

    # Extract text from the image using OCR
    extracted_text = pytesseract.image_to_string(image)

    # Convert the image to a base64-encoded string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare the data dictionary
    data_dict = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}",
        }
    }

    return extracted_text, data_dict
