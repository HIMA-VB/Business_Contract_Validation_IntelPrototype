from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
import torch
import torch.nn.functional as F
from compare_highlight import *
from transformers import AutoTokenizer, AutoModel
from text_classifier import extract_text_from_pdf, classify_clause, segregate_headings_clauses
from summarizer import compare_pdf
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class Condition(BaseModel):
    text: str

class ProcessedData(BaseModel):
    headings: List[str]
    contents: List[str]
    positions: List[dict]

# Function to compare sentence similarity
def compare_sentence_similarity(sentences1, sentences2):
    inputs = tokenizer(sentences1 + sentences2, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**inputs)
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1)
    similarities = []
    for i in range(len(sentences1)):
        sim = F.cosine_similarity(sentence_embeddings[i], sentence_embeddings[len(sentences1):], dim=1)
        similarities.append(sim)
    return similarities

@app.get("/")
async def root():
    return {"message": "Welcome to the Business Contract Validation API"}

@app.post("/upload_and_process_pdf/")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    pdf_text = extract_text_from_pdf(temp_file_path)
    headings, contents = separate_headings_and_content(pdf_text)
    positions = extract_text_and_positions_from_pdf(temp_file_path)

    os.unlink(temp_file_path)

    return ProcessedData(headings=headings, contents=contents, positions=positions)

@app.post("/classify_text/")
async def classify_text(text: str):
    type_pred, category_pred = classify_clause(text)
    return {"type": type_pred, "category": category_pred}

@app.post("/compare_headings/")
async def compare_headings_endpoint(headings1: List[str], headings2: List[str]):
    deviations = compare_headings(headings1, headings2)
    return deviations

@app.post("/compare_semantics/")
async def compare_semantics_endpoint(content1: List[str], content2: List[str]):
    deviations = compare_semantics(content1, content2, model, tokenizer)
    return deviations

@app.post("/highlight_deviations/")
async def highlight_deviations_endpoint(file: UploadFile = File(...), deviations: List[dict] = []):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    positions = extract_text_and_positions_from_pdf(temp_file_path)
    highlighted_pdf = highlight_deviations_in_pdf(temp_file_path, deviations, positions)

    output_path = "highlighted_" + file.filename
    highlighted_pdf.save(output_path)

    os.unlink(temp_file_path)

    return FileResponse(output_path, media_type="application/pdf", filename=output_path)

@app.post("/rule_based_search/")
async def rule_based_search(conditions: List[Condition], file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    text = extract_text_from_pdf(temp_file_path)
    sentences = text.split('\n')
    
    results = []
    for condition in conditions:
        similarities = compare_sentence_similarity([condition.text], sentences)
        for i, similarity in enumerate(similarities[0]):
            if similarity.item() >= 0.80:
                results.append({
                    "condition": condition.text,
                    "sentence": sentences[i],
                    "similarity_score": similarity.item()
                })

    os.unlink(temp_file_path)

    return results

@app.post("/summarize_pdfs/")
async def summarize_pdfs(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file2:
        temp_file1.write(await file1.read())
        temp_file2.write(await file2.read())
        temp_file1_path = temp_file1.name
        temp_file2_path = temp_file2.name

    summary1, summary2 = compare_pdf(temp_file1_path, temp_file2_path)

    os.unlink(temp_file1_path)
    os.unlink(temp_file2_path)

    return {"summary1": summary1, "summary2": summary2}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)