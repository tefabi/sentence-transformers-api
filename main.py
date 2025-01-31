from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from models import EmbeddingModel, ModelSettings

app = FastAPI()


@app.get("/health/")
async def health():
    return {"message": "Ok"}


class EmbeddingBody(BaseModel):
    model: EmbeddingModel = Field(default=EmbeddingModel.all_mpnet_base_v2)
    text: str


class EmbeddingResponse(BaseModel):
    text: str
    model: str
    embeddings: list[float]


@app.post("/embeddings/")
async def embeddings(data: EmbeddingBody) -> EmbeddingResponse:
    setting = ModelSettings.models[data.model]
    input_token_count = len(data.text) / 4
    if input_token_count > setting.max_sequence:
        errors = [
            {
                "loc": ("text"),
                "msg": f"The 'text' field length({len(data.text)}) must not be greater than {setting.max_sequence * 4} characters.",
                "type": "value_error",
            }
        ]
        raise RequestValidationError(errors)

    model = SentenceTransformer(data.model.value)
    embeddings = model.encode(data.text).tolist()

    return EmbeddingResponse(
        text=data.text, model=data.model.value, embeddings=embeddings
    )
