import chromadb
from sentence_transformers import SentenceTransformer
from splitter import *
import uuid

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="course_notes")

model = SentenceTransformer("all-MiniLM-L6-v2")


def add_document_to_vector_store(text : str, source : str = "user_input") -> list[dict]:
    chunks = split_by_size(text)

    docs = []
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source}-{uuid.uuid4()}-{i}"

        ids.append(chunk_id)
        docs.append(chunk)
        metadatas.append(
            {
                "source" : source,
                "chunk_index" : i
            }
        )

    embeddings = model.encode(docs).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas
    )

    result = []

    for i, chunk in enumerate(chunks):
       result.append({
            "id": ids[i],
            "content": chunk,
            "source": source
        })
       
    return result


def search_vector_store(query : str, top_k : int = 3) -> list[dict]:
    query_embeddings = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embeddings],
        n_results=top_k
    )

    contexts = []

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    for i in range(len(documents)):
        contexts.append({
                "id": ids[i],
                "content": documents[i],
                "source": metadatas[i]["source"],
                "chunk_index": metadatas[i]["chunk_index"]
            })
    
    return contexts


def list_all_chunks(limit : int = 20) -> list[dict]:
    result = collection.get(limit=limit)

    chunks = []

    ids = result.get("ids",[])
    documents = result.get("documents",[])
    metadatas = result.get("metadatas", [])
    
    for i in range(len(documents)):
        metadata = metadatas[i] or {}

        chunks.append({
            "id": ids[i],
            "content": documents[i],
            "source": metadata.get("source"),
            "chunk_index": metadata.get("chunk_index")
        })

    return chunks

def clear_vector_store() -> int:
    results = collection.get()
    ids = results.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return len(ids)
