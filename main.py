from fastapi import FastAPI
from pydantic import BaseModel
from llm_client import summarize_note
from splitter import split_by_paragraph

app = FastAPI()

class SplitRequest(BaseModel):
    text : str

class SummerizeRequest(BaseModel):
    text : str


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
