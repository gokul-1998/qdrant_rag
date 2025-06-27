
from sec import url,api_key
from qdrant_client import QdrantClient
from fastapi import FastAPI, Query, Response, UploadFile, status, HTTPException, Depends, APIRouter, Form, Depends, File, Header, Body

from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import List,Dict,Any,Union

from utils import (
    get_document_class,
    get_pdf_loader,
    get_text_splitter,
    get_prompt_template,
    get_sentence_transformer
)
import logging
logger=logging.getLogger(__name__)
from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-mpnet-base-v2"):  # Using a model with 768 dimensions
        self.model_name = model_name
        self._model = None
        self._dimension = 768  # Explicitly set expected dimension
    
    @property
    def model(self):
        if self._model is None:
            SentenceTransformer = get_sentence_transformer()
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def dimension(self):
        """Return the dimension of the embeddings."""
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False).tolist()
        # Validate dimensions
        if embeddings and len(embeddings[0]) != self._dimension:
            raise ValueError(f"Expected embedding dimension {self._dimension}, got {len(embeddings[0])}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if not text or not isinstance(text, str) or text.strip() == "":
            # Return a zero vector of the expected dimension for empty/invalid text
            return [0.0] * self._dimension
            
        try:
            # Ensure text is a string and not empty
            text = str(text).strip()
            if not text:
                return [0.0] * self._dimension
                
            # Get the embedding
            embeddings = self.model.encode([text], convert_to_tensor=False)
            
            # Handle case where encode returns a 2D array (which it should)
            if len(embeddings) > 0:
                embedding = embeddings[0].tolist()
            else:
                return [0.0] * self._dimension
                
            # Validate dimension
            if len(embedding) != self._dimension:
                logger.warning(f"Unexpected embedding dimension: {len(embedding)}. Expected: {self._dimension}")
                # Pad or truncate to match expected dimension
                if len(embedding) < self._dimension:
                    embedding = embedding + [0.0] * (self._dimension - len(embedding))
                else:
                    embedding = embedding[:self._dimension]
                    
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            logger.error(f"Input text was: {text}")
            # Return a zero vector of the expected dimension on error
            return [0.0] * self._dimension



class RegionalRagSystem:
    def __init__(self, collection_name: str = "test_rag"):
        # Initialize embeddings with a model that produces 768-dimensional vectors
        self.embeddings = CustomEmbeddings(model_name="all-mpnet-base-v2")
        self.collection_name = collection_name
        
        # Lazy-load RecursiveCharacterTextSplitter
        RecursiveCharacterTextSplitter = get_text_splitter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=url,
            api_key=api_key
        )
        
        # Create collection if it doesn't exist
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            # Get the vector size from the collection info
            vector_size = collection_info.config.params.vectors.size
            if vector_size != 768:
                # Delete and recreate collection with correct dimensions
                self.qdrant_client.delete_collection(self.collection_name)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            # Create collection if it doesn't exist
            error_message = str(e).lower()
            if (
                isinstance(e, UnexpectedResponse) and getattr(e, "status_code", None) == 404
            ) or "doesn't exist" in error_message or "not found" in error_message:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                )
            else:
                # Re-raise unexpected exceptions
                raise e
        
        # Ensure payload indexes for metadata filtering
        for field_name, field_schema in [
            ("metadata.team", "integer"),
            ("metadata.subteam_ids_str", "keyword"),
        ]:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
            except Exception:
                pass

    def _process_file(self, file: UploadFile) -> List:
        # Create a temporary file
        temp_file_path = f"temp_{secure_filename(file.filename)}"
        try:
            # Save the uploaded file to a temporary location
            with open(temp_file_path, "wb") as temp_file:
                content = file.file.read()
                temp_file.write(content)
                # Reset file pointer for potential reuse
                file.file.seek(0)
            
            # Process the file based on its type
            file_processor = FileProcessor(temp_file_path)
            content = file_processor.process()
            
            # Create a Document object
            Document = get_document_class()
            return [Document(page_content=content, metadata={})]
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def add_document(self, file: UploadFile, metadata: Dict[str, Any]):
        documents = self._process_file(file)
        for doc in documents:
            doc.metadata.update(metadata)
        splits = self.text_splitter.split_documents(documents)
        
        # Convert documents to Qdrant format
        points = []
        for i, split in enumerate(splits):
            embedding = self.embeddings.embed_query(split.page_content)
            points.append(
                models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "content": split.page_content,
                        "metadata": split.metadata
                    }
                )
            )
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(splits)
    
    def delete_document(self, s3_url: str):
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.s3_url",
                        match=models.MatchValue(value=s3_url)
                    )
                ]
            )
        )
        return True
        

    def query_by_metadata(self, query: str, metadata: Dict[str, Any], k: int = 4) -> List:
        embedding = self.embeddings.embed_query(query)
        
        # Build Qdrant filter
        qdrant_filter = None
        if metadata:
            filter_conditions = []
            for key, value in metadata.items():
                if key == "subteam_ids" and isinstance(value, list):
                    if value:
                        subteam_ids_str = ",".join(value)
                        filter_conditions.append(
                            models.FieldCondition(
                                key="metadata.subteam_ids_str",
                                match=models.MatchText(text=subteam_ids_str)
                            )
                        )
                else:
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )
            
            if filter_conditions:
                qdrant_filter = models.Filter(must=filter_conditions)
            print("filter conditions",filter_conditions)
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=qdrant_filter,
            limit=k
        )
        
        return [
            Document(
                page_content=hit.payload["content"],
                metadata=hit.payload["metadata"]
            )
            for hit in results
        ]

    def rag_response(self, query: str, metadata: Dict[str, Any] = None):
        try:
            # Increase k from 4 to 8
            docs = self.query_by_metadata(query, metadata, k=8) 
            print(f"found {len(docs)} results for the query")
            if not docs:
                return f"No info found for '{query}'"
            print(docs)
            print("before context")
            context = "\n\n".join([doc.page_content for doc in docs])
            # print(context)
            print(len(context))
            prompt = f"""
            Based on the following information, please answer the question.

            INFORMATION:
            {context}

            QUESTION: {query}

            ANSWER:
            """
            model_name = "llama3"
            api_url = os.getenv("OLLAMA_API_URL")
            api_key = os.getenv("OLLAMA_API_KEY")
            llm = OllamaAPI(model_name=model_name, api_url=api_url, api_key=api_key)
            return llm.generate_text(prompt)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_subteam_ids_based_on_query(self, query: Union[str, List[str]], metadata: Dict[str, Any] = None) -> list:
        """
        Get relevant subteam IDs based on a query by searching through the document store.
        
        Args:
            query: The search query (can be a string or list of strings)
            metadata: Optional metadata filters for the search
            
        Returns:
            Comma-separated string of subteam IDs, or an error message if no results found
        """
        try:
            # Convert list to string if needed
            if isinstance(query, list):
                query = " ".join(str(q) for q in query if q)
                
            if not query or not str(query).strip():
                logger.warning("Empty query received in get_subteam_ids_based_on_query")
                return "No query provided"
                
            query = str(query).strip()
            logger.info(f"Searching for subteams with query: '{query}', metadata: {metadata}")
            
            # Search for relevant documents
            docs = self.query_by_metadata(query, metadata or {}, k=5)
            
            if not docs:
                logger.info(f"No documents found for query: '{query}'")
                return f"No information found for '{query}'"
                
            # Log the top results for debugging
            logger.info(f"Found {len(docs)} relevant documents for query: '{query}'")
            for i, doc in enumerate(docs):  
                logger.info(f"Doc {i+1} metadata: {doc.metadata}")
            
            # Extract subteam_ids from all matching documents and count occurrences
            subteam_counter = {}
            for doc in docs:
                try:
                    subteam_ids = doc.metadata.get("subteam_ids_str", "")
                    if subteam_ids:
                        # Split the comma-separated string and count occurrences
                        for subteam_id in subteam_ids.split(','):
                            subteam_id = subteam_id.strip()
                            if subteam_id:
                                subteam_counter[subteam_id] = subteam_counter.get(subteam_id, 0) + 1
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Error processing document metadata: {e}")
                    continue
                    
            if not subteam_counter:
                logger.info(f"No subteam IDs found in results for query: '{query}'")
                return f"No relevant subteam information found for '{query}'"
                
            # Sort by frequency (descending) and get the top 3 most common subteams
            top_subteams = sorted(subteam_counter.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Extract just the subteam IDs for logging
            subteam_ids = [subteam_id for subteam_id, _ in top_subteams]
            logger.info(f"Returning subteams for query '{query}': {subteam_ids}")
            logger.debug("huha huha huha")
            return [i[0] for i in top_subteams]            
        except Exception as e:
            error_msg = f"Error determining subteams: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return error_msg


if __name__ == "__main__":
    rag_system = RegionalRagSystem()
    rag_system.add_document()
