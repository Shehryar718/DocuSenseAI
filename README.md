# DocuSenseAI

## Description

**DocuSenseAI** is an AI-powered tool designed to query and retrieve relevant documents across various file formats, including PDFs, text files, CSVs, Excel spreadsheets, and images.

### Supported Document Formats

- **PDF** (`.pdf`)
- **Text** (`.txt`)
- **CSV** (`.csv`)
- **Excel** (`.xlsx`)
- **Image** (`.png`, `.jpg`, `.jpeg`, `.gif`)

## Motivation

When working with textual and image data, I discovered that cosine similarity does not perform well for images. Even when using embedding models like CLIP for both images and text, the latent spaces differ significantly, leading to inaccurate similarity measures.

## Approach

For images, I use PyTesseract to extract text, followed by the OpenAI API to generate a description of the image. The embeddings of these descriptions are then stored in a vector database. A similar approach is applied to text documents, where a description is generated using the OpenAI API, and its embeddings are stored in the vector database.

## Metadata

Each record in the vector database contains the following metadata:

- **Type**
- **Description**
- **Content**
- **Path**

## Retrieval Process

The top K documents' metadata is incorporated into the chat history along with the system prompt for the OpenAI API. A retrieval prompt is then added, and the response includes the answer to the query as well as the path to the relevant document.

## Prerequisites

- **Python** 3.9+
- **Tesseract OCR** (system dependency for image text extraction)
- **OpenAI API key**

### Install Tesseract

**macOS:**

```bash
brew install tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev
```

**Windows:**

Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your PATH.

### Set up your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

Or export it directly:

```bash
export OPENAI_API_KEY=your-api-key-here
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

```python
from docusenseai import DocuSenseAI

dsa = DocuSenseAI()

# Add a document to the collection
dsa.add_document("my_docs", "path/to/report.pdf")

# Query the collection
response = dsa.query("my_docs", "What is the revenue?")
print(response)

# Delete the collection
dsa.delete_collection("my_docs")
```
