from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from llm_client import *
from splitter import split_by_paragraph
from rag_store import *
from vector_store import *

app = FastAPI()

class SplitRequest(BaseModel):
    text : str

class SummerizeRequest(BaseModel):
    text : str

class AdddocumentRequest(BaseModel):
    text : str
    source : str = "user_input"

class AskRequest(BaseModel):
    question : str
    top_k : int = 3



@app.get("/")
def root():
    return {"message": "helloapi"}

@app.post("/split")
def split_text(request: SplitRequest):
    return {"chunks": split_by_paragraph(request.text)}

@app.post("/summarize")
def summarize_text(request : SummerizeRequest):
    result = summarize_note(request.text)
    return result

# @app.post("/adddocument")
# def add_doc(request : AdddocumentRequest):
#     doc_chunks = add_document(request.text, request.source)

#     return {
#         "messsage" : "Document added",
#         "chunk_count" : len(doc_chunks),
#         "chunks" : doc_chunks
#     }

# @app.post("/ask")
# def ask(request : AskRequest):
#     contexts = search_chunks(request.question, request.top_k)
    
#     if len(contexts) == 0:
#         return {
#             "message" : "Nothing can be searched!\nPlease try another question or add more documents.",
#             "context" : []
#         }
    
#     result = answer_with_context(request.question, contexts)

#     return {
#         "answer" : result["answer"],
#         "used_chunks" : result.get("used_chunks", []),
#         "contexts" : contexts
#     }


@app.post("/adddocument")
def add_docu(request : AdddocumentRequest):
    doc_chunks = add_document_to_vector_store(request.text, request.source)

    return {
        "messsage" : "Document added",
        "chunk_count" : len(doc_chunks),
        "chunks" : doc_chunks
    }

@app.post("/ask")
def ask(request : AskRequest):
    contexts = search_vector_store(request.question, request.top_k)
    
    if len(contexts) == 0:
        return {
            "message" : "Nothing can be searched!\nPlease try another question or add more documents.",
            "context" : []
        }
    
    result = answer_with_context(request.question, contexts)

    return {
        "answer" : result["answer"],
        "used_chunks" : result.get("used_chunks", []),
        "contexts" : contexts
    }

@app.post("/upload_file")
async def upload_file(file : UploadFile = File(...)):
    filename = file.filename

    if not filename.endswith((".txt", ".md")):
        return {
            "error": "Only .txt and .md file are supported now."
        }
    
    content_bytes = await file.read()

    text = content_bytes.decode("utf-8")

    added_chunks = add_document_to_vector_store(
        text=text,
        source=filename
    )

    return{
        "message": "file uploaded and added to vector store",
        "filename": filename,
        "chunk_count": len(added_chunks),
        "chunks": added_chunks
    }

@app.get("/chunks")
def get_chunks(limit : int = 20):
    chunks = list_all_chunks(limit)

    return {
        "count" : len(chunks),
        "chunks" : chunks
    }
