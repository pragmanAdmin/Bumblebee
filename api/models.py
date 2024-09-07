from pydantic import BaseModel

class InferenceRequest(BaseModel):
    input_text: str

class InferenceResponse(BaseModel):
    prediction: str
