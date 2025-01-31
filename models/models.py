from pydantic import BaseModel, Field
from enum import Enum


class EmbeddingModel(str, Enum):
    all_mpnet_base_v2 = "all-mpnet-base-v2"
    multi_qa_mpnet_base_dot_v1 = "multi-qa-mpnet-base-dot-v1"
    all_distilroberta_v1 = "all-distilroberta-v1"
    all_MiniLM_L12_v2 = "all-MiniLM-L12-v2"
    multi_qa_distilbert_cos_v1 = "multi-qa-distilbert-cos-v1"
    all_MiniLM_L6_v2 = "all-MiniLM-L6-v2"
    multi_qa_MiniLM_L6_cos_v1 = "multi-qa-MiniLM-L6-cos-v1"
    paraphrase_multilingual_mpnet_base_v2 = "paraphrase-multilingual-mpnet-base-v2"
    paraphrase_albert_small_v2 = "paraphrase-albert-small-v2"
    paraphrase_multilingual_MiniLM_L12_v2 = "paraphrase-multilingual-MiniLM-L12-v2"
    paraphrase_MiniLM_L3_v2 = "paraphrase-MiniLM-L3-v2"
    distiluse_base_multilingual_cased_v1 = "distiluse-base-multilingual-cased-v1"
    distiluse_base_multilingual_cased_v2 = "distiluse-base-multilingual-cased-v2"


class Setting(BaseModel):
    max_sequence: int
    dimensions: int


class ModelSettings:
    models: dict[EmbeddingModel, Setting] = {
        EmbeddingModel.all_mpnet_base_v2: Setting(max_sequence=384, dimensions=768),
        EmbeddingModel.multi_qa_mpnet_base_dot_v1: Setting(
            max_sequence=512, dimensions=768
        ),
        EmbeddingModel.all_distilroberta_v1: Setting(max_sequence=512, dimensions=768),
        EmbeddingModel.all_MiniLM_L12_v2: Setting(max_sequence=256, dimensions=384),
        EmbeddingModel.multi_qa_distilbert_cos_v1: Setting(
            max_sequence=512, dimensions=768
        ),
        EmbeddingModel.all_MiniLM_L6_v2: Setting(max_sequence=256, dimensions=384),
        EmbeddingModel.multi_qa_MiniLM_L6_cos_v1: Setting(
            max_sequence=512, dimensions=384
        ),
        EmbeddingModel.paraphrase_multilingual_mpnet_base_v2: Setting(
            max_sequence=128, dimensions=768
        ),
        EmbeddingModel.paraphrase_albert_small_v2: Setting(
            max_sequence=256, dimensions=768
        ),
        EmbeddingModel.paraphrase_multilingual_MiniLM_L12_v2: Setting(
            max_sequence=128, dimensions=384
        ),
        EmbeddingModel.paraphrase_MiniLM_L3_v2: Setting(
            max_sequence=128, dimensions=384
        ),
        EmbeddingModel.distiluse_base_multilingual_cased_v1: Setting(
            max_sequence=128, dimensions=512
        ),
        EmbeddingModel.distiluse_base_multilingual_cased_v2: Setting(
            max_sequence=128, dimensions=512
        ),
    }
