from fastapi import FastAPI, UploadFile, File, Form
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List
from uuid import uuid4
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF

load_dotenv()

app = FastAPI()
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
model = SentenceTransformer("all-MiniLM-L6-v2")
collection_name = os.getenv("COLLECTION_NAME")

# Init collection
@app.on_event("startup")
def startup():
    if collection_name not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...), tags: str = Form("")):
    results = []
    for file in files:
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, file.filename)

        # Write the raw bytes to disk
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Extract text based on file type and prepare points list (one point per page for PDFs)
        points: List[PointStruct] = []
        if file.filename.lower().endswith(".pdf"):
            # Process each page separately so that every page becomes a distinct vector
            doc = fitz.open(upload_path)
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text()
                if not page_text.strip():
                    # Skip empty pages
                    continue
                embedding = model.encode(page_text).tolist()
                uid = str(uuid4())
                points.append(
                    PointStruct(
                        id=uid,
                        vector=embedding,
                        payload={
                            "filename": file.filename,
                            "tags": tags,
                            "page": page_number,
                            "text": page_text  # Store full page text
                        }
                    )
                )
                results.append({"id": uid, "filename": file.filename, "page": page_number})
            doc.close()
        else:
            # Non-PDF files – entire text as a single point (you can add custom chunking here if required)
            with open(upload_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            embedding = model.encode(text).tolist()
            uid = str(uuid4())
            points.append(
                PointStruct(
                    id=uid,
                    vector=embedding,
                    payload={
                        "filename": file.filename,
                        "tags": tags,
                        "text": text
                    }
                )
            )
            results.append({"id": uid, "filename": file.filename})

        # Clean up the temporary file
        os.remove(upload_path)

        # Upsert all points for this file in a single request
        if points:
            qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
    return {"uploaded": results}


@app.get("/search/")
def search(q: str):
    query_vector = model.encode(q).tolist()
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        with_payload=True
    )
    result= [{"id": r.id, "score": r.score, "payload": r.payload} for r in search_result]
    
    return {
        "total": len(result),
        "results": result
    }


@app.delete("/delete/{vector_id}")
def delete_vector(vector_id: str):
    qdrant.delete(
        collection_name=collection_name,
        points_selector={"points": [vector_id]}
    )
    return {"status": "deleted", "id": vector_id}


@app.get("/list/")
def list_all():
    # Get collection info to get total count
    collection_info = qdrant.get_collection(collection_name=collection_name)
    total = collection_info.points_count
    
    # Get the points with pagination
    scroll = qdrant.scroll(collection_name=collection_name, limit=100, with_payload=True)
    
    return {
        "total": total,
        "items": [
            {"id": p.id, "payload": p.payload} for p in scroll[0]
        ]
    }
