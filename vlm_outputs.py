from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class BinaryAnswer(str, Enum):
    YES = "YES"
    NO = "NO"

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

class BrandBinaryRecognitionOutput(BaseModel):
    answer: BinaryAnswer = Field(..., description="Binary decision: YES if the image likely represents the brand, otherwise NO")
    score: int = Field(..., ge=0, le=5, description="Confidence score between 0 and 5")

class BrandMulticlassRecognitionOutput(BaseModel):
    brand: BrandLabel = Field(..., description="Predicted brand label or NO_BRAND")
    score: int = Field(..., ge=0, le=5, description="Confidence score between 0 and 5")


class UnbrandingEvaluationOutput(BaseModel):
    image_1: Dict[str, Any]
    image_2: Dict[str, Any]
    pairwise_summary: Dict[str, Any]


class ConsistencyScore(BaseModel):
    score: float
    explanation: str

class VisualSimilarityEvaluationOutput(BaseModel):
    object_structure_consistency: ConsistencyScore
    color_texture_consistency: ConsistencyScore
    background_context_consistency: ConsistencyScore
    overall_visual_fidelity: ConsistencyScore
    final_similarity_score: float

class VisualSimilarityEvaluationOutputV2(BaseModel):
    explanation: str = Field(max_length=120)
    similarity_score: int
    
