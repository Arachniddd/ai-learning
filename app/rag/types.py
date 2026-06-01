from pydantic import BaseModel, Field
from typing import Optional

class Chunk(BaseModel):
    id : str
    content : str
    source : str
    chunk_index : int
    section : Optional[str] = None
    token_count : Optional[int] = None


class RetrieveChunk(Chunk):
    distance : Optional[float] = None
    score : Optional[float] = None
    rerank_score : Optional[float] = None