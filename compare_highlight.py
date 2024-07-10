import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein  # For calculating string similarity

# Function to extract text and its positions from PDF
def extract_text_and_positions_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text_positions = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_positions.append((span["text"], span["bbox"], page_num))
    return text_positions

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text_positions = extract_text_and_positions_from_pdf(pdf_file)
    text = "\n".join([item[0] for item in text_positions])
    return text

# Function to separate headings and content
def separate_headings_and_content(text):
    lines = text.split('\n')
    headings = []
    contents = []
    current_heading = None
    current_content = []

    for line in lines:
        if line.isupper() and line.strip():
            if current_heading:
                contents.append('\n'.join(current_content).strip())
                current_content = []
            current_heading = line.strip()
            headings.append(current_heading)
        else:
            current_content.append(line.strip())
    
    if current_heading:
        contents.append('\n'.join(current_content).strip())

    return headings, contents

# Function to embed texts using SentenceTransformer
def embed_texts(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

# Function to compare semantics and detect deviations
def compare_semantics(content1, content2, model, tokenizer, threshold=0.99):
    embeddings1 = embed_texts(content1, model, tokenizer)
    embeddings2 = embed_texts(content2, model, tokenizer)
    
    similarities = cosine_similarity(embeddings1, embeddings2)
    deviations = []

    for i in range(min(len(embeddings1), len(embeddings2))):
        sim_score = similarities[i, i]
        if sim_score < threshold:
            deviation_info = {
                'pdf1_diff': content1[i],
                'pdf2_diff': content2[i],
                'similarity_score': sim_score,
                'index1': i,
                'index2': i
            }
            deviations.append(deviation_info)
    
    return deviations

# Function to compare headings and detect differences
def compare_headings(headings1, headings2):
    heading_deviations = []

    for heading1 in headings1:
        if heading1 not in headings2:
            heading_deviations.append({'heading': heading1, 'deviation': 'PDF 2 missing this heading', 'pdf': 1})

    for heading2 in headings2:
        if heading2 not in headings1:
            heading_deviations.append({'heading': heading2, 'deviation': 'PDF 1 missing this heading', 'pdf': 2})

    return heading_deviations

# Function to highlight deviations in PDF
def highlight_deviations_in_pdf(pdf_file, deviations, positions):
    doc = fitz.open(pdf_file)

    for deviation in deviations:
        if 'heading' in deviation:
            diff_text = deviation['heading']
        else:
            diff_text = deviation['pdf1_diff'] if deviation.get('pdf', 1) == 1 else deviation['pdf2_diff']
        
        # Find the closest matching text region using Levenshtein distance
        best_match = find_best_match(diff_text, positions)

        if best_match is not None:
            text, bbox, page_num = best_match
            page = doc[page_num]
            highlight = page.add_highlight_annot(bbox)
            highlight.update()

    return doc

# Function to find the best matching text region
def find_best_match(target_text, positions):
    best_match = None
    best_score = float('inf')

    for text, bbox, page_num in positions:
        score = Levenshtein.distance(target_text.lower(), text.lower())  # Using Levenshtein distance for similarity
        if score < best_score:
            best_score = score
            best_match = (text, bbox, page_num)
    
    # Return the best match if similarity score is below a threshold
    if best_score < len(target_text) / 3:
        return best_match
    else:
        return None

