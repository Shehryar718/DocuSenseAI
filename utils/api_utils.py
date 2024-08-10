import os
from openai import OpenAI
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from utils.prompts import generation_prompt
from utils.parser import (
    get_image_data, 
    get_pdf_data, 
    get_excel_data, 
    get_csv_data, 
    get_txt_data
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key: str = os.environ.get('OPENAI_API_KEY')
client: OpenAI = OpenAI(api_key=api_key)

def generate_description(file_path: str, user_message: str = '') -> Tuple[str, str]:
    """
    Generates a description of the content within a file, using a conversational AI model to enhance the detail and relevance of the description.

    Parameters:
    -----------
    file_path : str
        The path to the file that needs to be described. The file can be of various formats such as PDF, text, CSV, Excel, or image.
    user_message : str, optional
        An additional user-provided message or context that can be appended to the conversation with the AI model to influence the generated description.

    Returns:
    --------
    Tuple[str, str]
        A tuple containing:
        - The generated description of the file content.
        - The raw content extracted from the file.

    Raises:
    -------
    ValueError:
        If the file format is not supported.
    
    Notes:
    ------
    The function supports the following file types:
        - PDF (.pdf)
        - Text (.txt)
        - CSV (.csv)
        - Excel (.xlsx)
        - Image (.png, .jpg, .jpeg, .gif, .JPG)
    
    Example:
    --------
    description, content = generate_description('/path/to/file.pdf')
    """
    chat_history = [{"role": "system", "content": generation_prompt}]
    file_content = ""

    if file_path.endswith('.pdf'):
        file_content = get_pdf_data(file_path)
        chat_history.append({"role": "user", "content": file_content})
    elif file_path.endswith('.txt'):
        file_content = get_txt_data(file_path)
        chat_history.append({"role": "user", "content": file_content})
    elif file_path.endswith('.csv'):
        file_content = get_csv_data(file_path)
        chat_history.append({"role": "user", "content": file_content})
    elif file_path.endswith('.xlsx'):
        file_content = get_excel_data(file_path)
        chat_history.append({"role": "user", "content": file_content})
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        extracted_text, image_data = get_image_data(file_path)
        text_data = {"type": "text", "text": extracted_text}
        chat_history.append({"role": "user", "content": [image_data, text_data]})
    else:
        raise ValueError(f"Unsupported file format: {file_path.split('.')[-1]}")

    if user_message:
        chat_history.append({"role": "user", "content": user_message})

    response = run_api(chat_history)
    
    return response, file_content

def run_api(messages: List[Dict[str, str]]) -> str:
    """
    Sends a series of messages to the GPT-4o-mini model via the OpenAI API and retrieves the generated response.

    Parameters:
    -----------
    messages : List[Dict[str, str]]
        A list of dictionaries representing the conversation history. Each dictionary should have the following keys:
        - 'role': Specifies the role in the conversation ('system', 'user', or 'assistant').
        - 'content': The content of the message.

    Returns:
    --------
    str
        The content of the AI-generated response.

    Raises:
    -------
    openai.error.OpenAIError:
        If there is an error in the API request.
    
    Example:
    --------
    response = run_api([{"role": "system", "content": "Describe the content of the document."}, {"role": "user", "content": "The document is about AI."}])
    """
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Handle or log the error as needed
        raise RuntimeError(f"API request failed: {str(e)}")
