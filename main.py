from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from ai_engine import analyze_resume
import io
from typing import Optional

app = FastAPI(title="AI Resume Reviewer Backend", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Resume Reviewer Backend Running 🚀"}

@app.post("/analyze")
async def analyze_resume_api(
    resume: Optional[UploadFile] = File(None),
    question: str = Form(...)
):
    try:
        text = None

        if resume:
            pdf_bytes = await resume.read()
            text = ""
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if not text.strip():
                return {"status": "error", "error": "No text found in resume PDF"}

        ai_analysis = analyze_resume(text, question)

        if "error" in ai_analysis:
            return {"status": "error", "error": ai_analysis["error"]}

        return {"status": "success", "data": ai_analysis}

    except Exception as e:
        return {"status": "error", "error": f"Server Error: {str(e)}"}