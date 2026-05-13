from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents import medical_crew

app = FastAPI(title="Medical Assistant API")

class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Medical Chat Assistant API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = medical_crew.kickoff(inputs={'query':request.query})
        answer = response.raw

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
