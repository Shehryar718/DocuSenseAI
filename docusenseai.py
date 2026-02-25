import os
import uuid
import logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from utils.api_utils import run_api, generate_description
from utils.prompts import retrieval_prompt
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

DEFAULT_VECTORSTORE = "instance/VECTORSTORE"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384


class DocuSenseAI:
    """
    AI-powered document query and retrieval system.

    Parameters:
    -----------
    vectorstore_path : str, optional
        Path to the Milvus vector store. Default is "instance/VECTORSTORE".
    embedding_model : str, optional
        Name of the SentenceTransformer model. Default is "all-MiniLM-L6-v2".
    dimension : int, optional
        Embedding vector dimension. Must match the model. Default is 384.

    Example:
    --------
    dsa = DocuSenseAI()
    dsa.create_collection("my_docs")
    dsa.add_document("my_docs", "report.pdf")
    response = dsa.query("my_docs", "What is the revenue?")
    """

    def __init__(
        self,
        vectorstore_path: str = DEFAULT_VECTORSTORE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_DIMENSION
    ):
        os.makedirs(os.path.dirname(vectorstore_path) or ".", exist_ok=True)
        self.dimension = dimension
        self.model = SentenceTransformer(embedding_model)
        try:
            self.vdb_client = MilvusClient(vectorstore_path)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus at '{vectorstore_path}': {e}")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Generates an embedding vector from the input text."""
        return self.model.encode(text).tolist()

    def create_collection(self, collection_name: str) -> None:
        """Creates a collection in Milvus if it does not already exist."""
        try:
            if not self.vdb_client.has_collection(collection_name=collection_name):
                self.vdb_client.create_collection(
                    collection_name=collection_name,
                    dimension=self.dimension
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection '{collection_name}': {e}")

    def add_document(self, collection_name: str, path: str) -> None:
        """
        Adds a document to the Milvus collection.

        Parameters:
        -----------
        collection_name : str
            The name of the collection to add the document to.
        path : str
            The path to the document file.
        """
        if not self.vdb_client.has_collection(collection_name=collection_name):
            self.create_collection(collection_name)

        description, text = generate_description(path)
        vector = self._get_text_embedding(description)
        idx = uuid.uuid4().int % (2**63)

        metadata = {
            "type": os.path.splitext(path)[1].lstrip('.').lower(),
            "description": description,
            "content": text,
            "path": path
        }

        data = [{
            "id": idx,
            "vector": vector,
            "metadata": metadata
        }]

        try:
            self.vdb_client.insert(
                collection_name=collection_name,
                data=data,
                timeout=120
            )
        except Exception as e:
            raise RuntimeError(f"Failed to insert document '{path}' into '{collection_name}': {e}")

    def delete_collection(self, collection_name: str) -> None:
        """Deletes a collection from Milvus."""
        try:
            if self.vdb_client.has_collection(collection_name=collection_name):
                self.vdb_client.drop_collection(collection_name=collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection '{collection_name}': {e}")

    def retrieve_document(
        self,
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
        if not self.vdb_client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist. Create it and add documents first.")

        vector = self._get_text_embedding(text)

        try:
            results = self.vdb_client.search(
                collection_name=collection_name,
                data=[vector],
                output_fields=["metadata"],
                limit=top_k
            )
        except Exception as e:
            raise RuntimeError(f"Failed to search collection '{collection_name}': {e}")

        return results[0] if results else []

    def query(
        self,
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
            Whether to log the search results, by default False.

        Returns:
        --------
        str
            The response from the LLM.
        """
        search_results = self.retrieve_document(
            collection_name=collection_name,
            text=user_query,
            top_k=top_k
        )

        document_texts = [
            f"Document {idx + 1}:\n"
            f"  Description: {result['entity']['metadata']['description']}\n"
            f"  Content: {result['entity']['metadata']['content']}\n"
            f"  Path: {result['entity']['metadata']['path']}"
            for idx, result in enumerate(search_results)
        ]

        if verbose:
            for result in search_results:
                logger.info(result)

        documents_block = "\n\n".join(document_texts)
        system_message = (
            "You are an AI that answers questions based on document content.\n\n"
            f"{retrieval_prompt.strip()}\n\n"
            f"--- Retrieved Documents ---\n{documents_block}"
        )

        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]

        response = run_api(conversation)
        return response
