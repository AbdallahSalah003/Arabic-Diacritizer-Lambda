from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from .infer import infer, diacritizer
from mangum import Mangum


class Request(BaseModel):
    text: str 

app = FastAPI()
MAX_TEXT_LENGTH = 128

def is_arabic_text(text: str) -> bool:
    arabic_chars = 0
    total_letters = 0

    for ch in text:
        if ch.isalpha():
            total_letters += 1
            if '\u0600' <= ch <= '\u06FF' or \
               '\u0750' <= ch <= '\u077F' or \
               '\u08A0' <= ch <= '\u08FF':
                arabic_chars += 1

    if total_letters == 0:
        return False

    return (arabic_chars / total_letters) > 0.7

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
        response["warning"] = "Please enter any Arabic text"
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