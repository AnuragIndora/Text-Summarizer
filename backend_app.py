from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
import os
from src.textSummarizer.pipeline.predication import TextSummarizer
import uvicorn
from pydantic import BaseModel


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Text Summarizer API!"}

@app.get("/train", tags=["Training Process"])
async def training():
    
    try:
        os.system("python main.py")
        return PlainTextResponse("Training Successful!")
    except Exception as e:
        return PlainTextResponse(f"Error Occurred: {e}")
    

# Define the request body model
class SummarizationRequest(BaseModel):
    input_text: str

@app.post("/summarize")
async def summarize_route(request: SummarizationRequest):
    try:
        ts = TextSummarizer()
        summary = ts.summarize(request.input_text)
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
