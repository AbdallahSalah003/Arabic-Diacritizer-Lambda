from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from .infer import infer, diacritizer
from mangum import Mangum
import re

class Request(BaseModel):
    text: str 

app = FastAPI()
MAX_TEXT_LENGTH = 128

ARABIC_PATTERN = re.compile(
    r'^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]+$' # Arabic letters + Arabic diacritics + Arabic digits + spaces
)

def is_arabic_text(text: str) -> bool:
    if not text.strip():
        return False

    return bool(ARABIC_PATTERN.fullmatch(text))

@app.get("/")
def health():
    return {"message": "Healthy!"}


@app.post("/predict")
def predict(request: Request):

    response = {
        "warning": None,
        "diacritized_text": None
    }
    text = request.text

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    if not is_arabic_text(text):
        response["warning"] = "Please enter only Arabic text"
        return response
    
    if len(diacritizer.extract_chars_and_labels(text)) > MAX_TEXT_LENGTH:
        response["warning"] = (
            f"Max length exceeded! Diacritizing only first {MAX_TEXT_LENGTH} chars."
        )

    try:
        response["diacritized_text"] = infer(text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response



lambda_handler = Mangum(app)