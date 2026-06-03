from fastapi import APIRouter, UploadFile, File
from app.api.schemas.document import SplitRequest, SummarizeRequest, AddDocumentRequest
from app.rag.splitter import split_by_paragraph
from app.rag.vector_store import (
    add_document_to_vector_store,
    list_all_chunks,
    clear_vector_store,
)
from app.llm.client import summarize_note


router = APIRouter()

@router.post("/split")
def split_text(request: SplitRequest):
    return {"chunks": split_by_paragraph(request.text)}


@router.post("/summarize")
def summarize_text(request: SummarizeRequest):
    result = summarize_note(request.text)
    return result


@router.post("/adddocument")
def add_document(request: AddDocumentRequest):
    doc_chunks = add_document_to_vector_store(request.text, request.source)
    return {
        "message": "Document added",
        "chunk_count": len(doc_chunks),
        "chunks": doc_chunks,
    }


@router.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename

    if not filename.endswith((".txt", ".md")):
        return {"error": "Only .txt and .md files are supported now."}

    content_bytes = await file.read()
    text = content_bytes.decode("utf-8")

    added_chunks = add_document_to_vector_store(
        text=text,
        source=filename,
    )

    return {
        "message": "file uploaded and added to vector store",
        "filename": filename,
        "chunk_count": len(added_chunks),
        "chunks": added_chunks,
    }


@router.get("/chunks")
def get_chunks(limit: int = 20):
    chunks = list_all_chunks(limit)
    return {
        "count": len(chunks),
        "chunks": chunks,
    }


@router.delete("/chunks")
def delete_chunks():
    num = clear_vector_store()
    return {"deleted_count": num}
