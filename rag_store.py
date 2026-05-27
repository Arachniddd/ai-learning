from splitter import *

chunks = []

def add_document(text : str, source : str = "user_input") -> list[dict]:
    global chunks

    new_chunks = split_by_paragraph(text)
    start_id = len(chunks)

    result = []

    for i, chunk in enumerate(new_chunks):
        item = {
            "id" : start_id + i,
            "content" : chunk,
            "source" : source
        }
        result.append(item)
        chunks.append(item)
    
    return result

def search_chunks(query : str, top_k : int = 3) -> list[dict]:
    scored = []

    query_words = set(query.lower().split())

    for chunk in chunks:
        chunk_words = set(chunk["content"].lower().split())
        score = len(query_words & chunk_words)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunk for score, chunk in scored[:top_k]]