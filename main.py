import base64
import io
from typing import Dict, Optional

import fitz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="PDF Text Extraction API",
    version="1.0.0",
    description="Professional PDF text extraction service with multi-language support.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractionRequest(BaseModel):
    content: str = Field(..., description="Base64 encoded PDF content")
    allPages: bool = Field(True, description="Extract all pages if true, otherwise only the first page")


class ExtractionResponse(BaseModel):
    success: bool
    totalPages: int
    text: Dict[str, str]


class RootResponse(BaseModel):
    success: bool
    message: str


def normalize_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.rstrip()
        if stripped:
            cleaned.append(stripped)
        elif cleaned and cleaned[-1] != "":
            cleaned.append("")
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    return "\n".join(cleaned)


def extract_page_text(page: fitz.Page) -> str:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_IMAGES)["blocks"]
    fragments = []
    for block in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        if block["type"] == 0:
            for line in block.get("lines", []):
                spans_text = ""
                for span in line.get("spans", []):
                    spans_text += span.get("text", "")
                if spans_text.strip():
                    fragments.append(spans_text)
            fragments.append("")
    return normalize_text("\n".join(fragments))


@app.get("/", response_model=RootResponse)
async def root():
    return RootResponse(
        success=True,
        message="Use /v1/extract via POST method to extract text from PDF.",
    )


@app.post("/v1/extract", response_model=ExtractionResponse)
async def extract_text(request: ExtractionRequest):
    try:
        pdf_bytes = base64.b64decode(request.content, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoded content.")

    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        document = fitz.open(stream=pdf_stream, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=422, detail="Unable to parse the provided content as a valid PDF.")

    total_pages = document.page_count

    if total_pages == 0:
        document.close()
        raise HTTPException(status_code=422, detail="The provided PDF contains no pages.")

    text_output: Dict[str, str] = {}

    try:
        if request.allPages:
            for page_number in range(total_pages):
                page = document.load_page(page_number)
                extracted = extract_page_text(page)
                text_output[str(page_number + 1)] = extracted
        else:
            page = document.load_page(0)
            extracted = extract_page_text(page)
            text_output["1"] = extracted
    except Exception:
        document.close()
        raise HTTPException(status_code=500, detail="An error occurred during text extraction.")
    finally:
        document.close()

    return ExtractionResponse(
        success=True,
        totalPages=total_pages,
        text=text_output,
    )
