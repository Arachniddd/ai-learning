import chromadb
from sentence_transformers import SentenceTransformer
from splitter import split_by_paragraph

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="course_notes")

model = SentenceTransformer("all-MiniLM-L6-v2")


def add_document_to_vector_store(text : str, source : str = "user_input") -> list[dict]:
    chunks = split_by_paragraph(text)

    docs = []
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source}-{i}"

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