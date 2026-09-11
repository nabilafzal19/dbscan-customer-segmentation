from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.predictor import predictor


app = FastAPI(
    title="DBSCAN Customer Segmentation API",
    description=(
        "Production API for customer segmentation "
        "using DBSCAN."
    ),
    version="1.0.0"
)


class CustomerRequest(BaseModel):

    age: float = Field(
        ...,
        gt=0,
        le=120,
        description="Customer age"
    )

    annual_income: float = Field(
        ...,
        gt=0,
        description="Annual income in thousands"
    )

    spending_score: float = Field(
        ...,
        ge=1,
        le=100,
        description="Customer spending score"
    )


class PredictionResponse(BaseModel):

    cluster: int

    segment: str

    distance_to_nearest_core: float


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "DBSCAN",
        "eps": predictor.eps,
        "min_samples": predictor.min_samples
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_customer(
    customer: CustomerRequest
):

    return predictor.predict(
        age=customer.age,
        annual_income=customer.annual_income,
        spending_score=customer.spending_score
    )