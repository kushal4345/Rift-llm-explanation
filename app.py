import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # ✅ ADDED
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from sarvamai import SarvamAI
import uvicorn

# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_SUBSCRIPTION_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

if not SARVAM_API_KEY:
    raise RuntimeError("SARVAM_API_SUBSCRIPTION_KEY not set")

# ==========================================
# CONFIGURE GEMINI (STABLE MODEL)
# ==========================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-3-flash-preview")

# ==========================================
# CONFIGURE SARVAM
# ==========================================

sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# ==========================================
# FASTAPI INIT
# ==========================================

app = FastAPI(
    title="Pharmacogenomics Explanation API",
    version="4.0.0"
)

# ==========================================
# CORS CONFIGURATION  ✅ ADDED
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (safe for hackathon/testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# REQUEST MODEL
# ==========================================

class ReportRequest(BaseModel):
    drug: str
    gene: str
    phenotype: str
    risk_label: str
    severity: str
    confidence_score: float
    cpic_level: str
    preferred_language: str
    tone: str = "formal"
    numerals_format: str = "international"
    speaker_gender: str = "Male"

# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are a pharmacogenomics clinical explanation engine.

STRICT RULES:
- Do NOT change risk label.
- Do NOT modify dosage recommendation.
- Do NOT hallucinate genes or drugs.
- Do NOT override CPIC evidence.
- Keep gene symbols unchanged.

LENGTH RULES:
- Clinician Summary: max 1 lines
- Patient Explanation: max 1 lines
- Mechanism Explanation: max 1 lines
- Monitoring Advice: max 1 lines
- Use concise medical language.
- Avoid repetition.

Generate structured sections:

Clinician Summary:
Patient Explanation:
Mechanism Explanation:
Monitoring Advice:
"""


# ==========================================
# CLEAN TEXT BEFORE TRANSLATION
# ==========================================

def clean_text_for_translation(text: str) -> str:
    text = text.replace("```", "")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ==========================================
# LLM FUNCTION
# ==========================================

def generate_llm_explanation(data: ReportRequest):

    prompt = f"""
{SYSTEM_PROMPT}

Drug: {data.drug}
Gene: {data.gene}
Phenotype: {data.phenotype}
Risk Label: {data.risk_label}
Severity: {data.severity}
Confidence Score: {data.confidence_score}
CPIC Level: {data.cpic_level}
"""

    response = gemini_model.generate_content(prompt)

    if not response or not response.text:
        raise Exception("Gemini returned empty response")

    return response.text

# ==========================================
# SAFE TRANSLATION FUNCTION
# ==========================================

def translate_text(text, target_language):

    try:
        cleaned_text = clean_text_for_translation(text)
        max_chunk_size = 500
        translated_chunks = []

        for i in range(0, len(cleaned_text), max_chunk_size):
            chunk = cleaned_text[i:i+max_chunk_size]
            print(f"Translating chunk of length {len(chunk)} to {target_language}")

            response = sarvam_client.text.translate(
                input=chunk,
                source_language_code="en-IN",
                target_language_code=target_language
            )

            if hasattr(response, "translated_text"):
                translated_chunks.append(response.translated_text)
            else:
                translated_chunks.append(str(response))

        return "\n".join(translated_chunks)

    except Exception as e:
        print("Sarvam FULL Error Details:", str(e))
        return None

# ==========================================
# API ENDPOINT
# ==========================================

@app.post("/generate-report")
def generate_report(data: ReportRequest):

    try:
        english_output = generate_llm_explanation(data)

        result = {
            "clinician_en": english_output,
            "patient": {
                "en-IN": english_output
            }
        }

        if data.preferred_language != "en-IN":

            translated = translate_text(
                english_output,
                data.preferred_language
            )

            if translated:
                result["patient"][data.preferred_language] = translated
            else:
                result["translation_error"] = "Translation failed"

        return {
            "status": "success",
            "drug": data.drug,
            "gene": data.gene,
            "risk_label": data.risk_label,
            "severity": data.severity,
            "confidence_score": data.confidence_score,
            "llm_generated_explanation": result
        }

    except Exception as e:
        print(f"Error in generate_report: {type(e).__name__}({e})")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {"status": "running"}

# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
