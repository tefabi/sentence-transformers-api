from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

@app.get('/health/')
async def health():
    return {'message': "Ok"}


class EmbeddingBody(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    text: str
    embeddings: list[int]


@app.post("/embeddings/")
async def embeddings(data: EmbeddingBody) -> EmbeddingResponse:
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(data.text).tolist()

    print(len(embeddings))
    return EmbeddingResponse(text=data.text, embeddings=embeddings)