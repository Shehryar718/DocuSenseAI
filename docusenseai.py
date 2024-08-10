import os
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from utils.api_utils import run_api, generate_description
from utils.prompts import retrieval_prompt

os.makedirs("instance", exist_ok=True)
VECTORSTORE = "instance/VECTORSTORE"
EMBEDDIGS_MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384

# Model and database setup
model = SentenceTransformer(EMBEDDIGS_MODEL_NAME)
vdb_client = MilvusClient(VECTORSTORE)

def get_text_embedding(text, model):
    return model.encode(text).tolist()

def create_collection(collection_name):
    if not vdb_client.has_collection(collection_name=collection_name):
        vdb_client.create_collection(
            collection_name=collection_name, 
            dimension=DIMENSION, 
        )

def add_document(collection_name, path):
    if not vdb_client.has_collection(collection_name=collection_name):
        create_collection(collection_name)
    
    description, text = generate_description(path)
    vector = get_text_embedding(description, model)
    idx = vdb_client.get_collection_stats(collection_name=collection_name)['row_count']

    metadata = {
        "type": path.split('.')[-1],
        "description": description,
        "content": text,
        "path": path
    }

    data = [{
        "id": idx,
        "vector": vector,
        "metadata": metadata
    }]

    vdb_client.insert(
        collection_name=collection_name, 
        data=data, 
        timeout=120
    )

def delete_collection(collection_name):
    if vdb_client.has_collection(collection_name=collection_name):
        vdb_client.drop_collection(collection_name=collection_name)

def retrieve_document(collection_name, text, top_k=3):
    if not vdb_client.has_collection(collection_name=collection_name):
        create_collection(collection_name)

    vector = get_text_embedding(text, model)

    results = vdb_client.search(
        collection_name=collection_name, 
        data=[vector],
        output_fields=["metadata"],
        limit=top_k
    )

    return results[0]

def retrieve_information_from_document(text, collection_name, top_k=3, verbose=False):
    search_results = retrieve_document(collection_name, text, top_k=top_k)
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
    conversation = [{"role": "system", "content": "You are an AI that answers questions based on document content."}]
    conversation.append({"role": "user", "content": text})
    
    # Add the content from the top-k documents
    conversation.extend([{"role": "system", "content": f"Document {idx + 1}: {doc_text}"} for idx, doc_text in enumerate(document_texts)])
    
    # Ask the LLM to choose the best matching document and answer the query
    conversation.append({"role": "user", "content": retrieval_prompt})

    response = run_api(conversation)
    return response

# Main function for conversational query
def query(collection_name, user_query, top_k=3, verbose=False):
    # Retrieves the relevant documents
    retrieved_info = retrieve_information_from_document(user_query, collection_name, top_k, verbose)
    
    return retrieved_info

