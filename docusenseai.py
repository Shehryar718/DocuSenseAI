import os
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from utils.api_utils import run_api, generate_description
from utils.prompts import retrieval_prompt
from typing import List, Dict, Union

# Directory and file setup
os.makedirs("instance", exist_ok=True)
VECTORSTORE = "instance/VECTORSTORE"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384

# Initialize model and database client
model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
vdb_client = MilvusClient(VECTORSTORE)

def get_text_embedding(text: str, model: SentenceTransformer) -> List[float]:
    """
    Generates an embedding vector from the input text using the provided model.

    Parameters:
    -----------
    text : str
        The input text to be encoded into an embedding vector.
    model : SentenceTransformer
        The model used to generate the embedding. It should have an `encode` method
        that takes a string input and returns an embedding vector.

    Returns:
    --------
    List[float]
        The embedding vector as a list of floats.

    Example:
    --------
    embedding = get_text_embedding("Hello, world!", model)
    """
    embedding = model.encode(text)
    return embedding.tolist()

def create_collection(collection_name: str) -> None:
    """
    Creates a collection in Milvus if it does not already exist.

    Parameters:
    -----------
    collection_name : str
        The name of the collection to be created.

    Returns:
    --------
    None
    """
    if not vdb_client.has_collection(collection_name=collection_name):
        vdb_client.create_collection(
            collection_name=collection_name, 
            dimension=DIMENSION
        )

def add_document(collection_name: str, path: str) -> None:
    """
    Adds a document to the Milvus collection.

    Parameters:
    -----------
    collection_name : str
        The name of the collection to add the document to.
    path : str
        The path to the document file.

    Returns:
    --------
    None
    """
    if not vdb_client.has_collection(collection_name=collection_name):
        create_collection(collection_name)
    
    description, text = generate_description(path)
    vector = get_text_embedding(description, model)
    idx = vdb_client.get_collection_stats(collection_name=collection_name)['row_count']

    metadata = {
        "type": path.split('.')[-1],  # file type
        "description": description,
        "content": text,
        "path": path
    }

    data = [{
        "id": idx,  # document ID
        "vector": vector,
        "metadata": metadata
    }]

    vdb_client.insert(
        collection_name=collection_name, 
        data=data, 
        timeout=120
    )

def delete_collection(collection_name: str) -> None:
    """
    Deletes a collection from Milvus.

    Parameters:
    -----------
    collection_name : str
        The name of the collection to delete.

    Returns:
    --------
    None
    """
    if vdb_client.has_collection(collection_name=collection_name):
        vdb_client.drop_collection(collection_name=collection_name)

def retrieve_document(
    collection_name: str,
    text: str,
    top_k: int = 3
) -> List[Dict[str, Union[str, int]]]:
    """
    Retrieves documents from a Milvus collection based on a text query.

    Parameters:
    -----------
    collection_name : str
        The name of the collection to retrieve documents from.
    text : str
        The query text to search for in the documents.
    top_k : int, optional
        The number of top documents to retrieve. Default is 3.

    Returns:
    --------
    List[Dict[str, Union[str, int]]]
        A list of dictionaries containing the metadata of the retrieved documents.
    """
    if not vdb_client.has_collection(collection_name=collection_name):
        create_collection(collection_name)

    vector = get_text_embedding(text, model)

    results = vdb_client.search(
        collection_name=collection_name, 
        data=[vector],
        output_fields=["metadata"],
        limit=top_k
    )

    return results[0] if results else []

def retrieve_information_from_document(
    text: str,
    collection_name: str,
    top_k: int = 3,
    verbose: bool = False
) -> str:
    """
    Retrieves information from documents based on a text query.

    Parameters:
    -----------
    text : str
        The query text to search for in the documents.
    collection_name : str
        The name of the collection to retrieve documents from.
    top_k : int, optional
        The number of top documents to retrieve. Default is 3.
    verbose : bool, optional
        Whether to print the retrieved documents. Default is False.

    Returns:
    --------
    str
        The response from the LLM.
    """
    search_results = retrieve_document(
        collection_name=collection_name,
        text=text,
        top_k=top_k
    )
    
    document_texts = [
        f"""
        Description: {result['entity']['metadata']['description']}
        Content: {result['entity']['metadata']['content']}
        Path: {result['entity']['metadata']['path']}
        """
        for result in search_results
    ]
    
    if verbose:
        for result in search_results:
            print(result)

    # Build conversation content with relevant document snippets
    conversation = [
        {"role": "system", "content": "You are an AI that answers questions based on document content."},
        {"role": "user", "content": text}
    ]
    conversation.extend([
        {"role": "system", "content": f"Document {idx + 1}: {doc_text}"}
        for idx, doc_text in enumerate(document_texts)
    ])

    # Ask the LLM to choose the best matching document and answer the query
    conversation.append({"role": "user", "content": retrieval_prompt})

    response = run_api(conversation)
    return response

def query(
    collection_name: str,
    user_query: str,
    top_k: int = 3,
    verbose: bool = False
) -> str:
    """
    Retrieves the relevant documents and returns the response from the LLM.

    Parameters:
    -----------
    collection_name : str
        Name of the collection to retrieve documents from.
    user_query : str
        The query string from the user.
    top_k : int, optional
        The number of top documents to retrieve, by default 3.
    verbose : bool, optional
        Whether to print the search results, by default False.

    Returns:
    --------
    str
        The response from the LLM.
    """
    retrieved_info = retrieve_information_from_document(
        text=user_query,
        collection_name=collection_name,
        top_k=top_k,
        verbose=verbose
    )
    
    return retrieved_info
