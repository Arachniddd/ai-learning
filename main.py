from fastapi import FastAPI
from pydantic import BaseModel

from splitter import split_by_paragraph


app = FastAPI()


class SplitRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "helloapi"}


@app.post("/split")
def split_text(request: SplitRequest):
    return {"chunks": split_by_paragraph(request.text)}
