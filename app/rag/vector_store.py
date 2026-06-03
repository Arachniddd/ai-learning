import chromadb
from sentence_transformers import SentenceTransformer
from app.rag.splitter import *
from app.models.chunk import Chunk, RetrieveChunk

client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_or_create_collection(name="knowledge_base")

model = SentenceTransformer("all-MiniLM-L6-v2")

def distance_to_score(distance: float) -> float:
    return round(1 / (1 + distance), 4)


def build_metadata(chunk: Chunk) -> dict:
    metadata = {
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "section": chunk.section,
        "token_count": chunk.token_count,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def add_document_to_vector_store(text : str, source : str = "user_input") -> list[Chunk]:
    chunks = split_text(text=text, source=source)

    docs = []
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(chunk.id)
        docs.append(chunk.content)
        metadatas.append(build_metadata(chunk))

    embeddings = model.encode(docs).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return chunks


def search_vector_store(query : str, top_k : int = 3) -> list[RetrieveChunk]:
    query_embeddings = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embeddings],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(documents)):
        metadata = metadatas[i] or {}
        distance = distances[i]

        contexts.append(
            RetrieveChunk(
                id=ids[i],
                content=documents[i],
                source=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", i),
                section=metadata.get("section"),
                token_count=metadata.get("token_count"),
                distance=distance,
                score=distance_to_score(distance),
            )
        )
    
    return contexts


def list_all_chunks(limit : int = 20) -> list[Chunk]:
    result = collection.get(limit=limit)

    chunks = []

    ids = result.get("ids",[])
    documents = result.get("documents",[])
    metadatas = result.get("metadatas", [])
    
    for i in range(len(documents)):
        metadata = metadatas[i] or {}

        chunks.append(
            Chunk(
                id=ids[i],
                content=documents[i],
                source=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", i),
                section=metadata.get("section"),
                token_count=metadata.get("token_count"),
            )
        )

    return chunks

def clear_vector_store() -> int:
    results = collection.get()
    ids = results.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return len(ids)
