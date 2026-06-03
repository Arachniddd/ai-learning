from pydantic import BaseModel


class SplitRequest(BaseModel):
    text: str


class SummarizeRequest(BaseModel):
    text: str


class AddDocumentRequest(BaseModel):
    text: str
    source: str = "user_input"
