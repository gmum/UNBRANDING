from enum import Enum

from pydantic import BaseModel, Field


class BrandLabel(str, Enum):
    ADIDAS = "ADIDAS"
    APPLE = "APPLE"
    AUDI = "AUDI"
    BMW = "BMW"
    COCA_COLA = "COCA_COLA"
    EMIRATES = "EMIRATES"
    MCDONALDS = "MCDONALDS"
    MERCEDES = "MERCEDES"
    MONSTER = "MONSTER"
    NIKE = "NIKE"
    PUMA = "PUMA"
    SINGAPORE_AIRLINES = "SINGAPORE_AIRLINES"
    NO_BRAND = "NO_BRAND"


class BrandMulticlassRecognitionOutput(BaseModel):
    brand: BrandLabel = Field(..., description="Predicted brand label or NO_BRAND")
    score: int = Field(..., ge=0, le=5, description="Confidence score between 0 and 5")


class VisualSimilarityEvaluationOutput(BaseModel):
    explanation: str = Field(max_length=120)
    similarity_score: float = Field(..., ge=0, le=10)
