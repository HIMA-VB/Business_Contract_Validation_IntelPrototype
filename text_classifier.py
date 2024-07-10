# text_classifier.py
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier
import fitz  # PyMuPDF for PDF processing
from io import BytesIO
import re

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Extended training data with main/sub-clause categorization
data = {
    'text': [
        "The price of the product is $1000.",
        "Delivery will be completed within 30 days.",
        "Payment must be made within 15 days of invoice.",
        "If the product is damaged, it will be replaced.",
        "The warranty period is 1 year from the date of purchase.",
        "This contract is governed by the laws of California.",
        "In the event of a force majeure, the obligations may be suspended.",
        "The company shall indemnify the client against any losses.",
        "Intellectual property rights shall remain with the creator.",
        "Any disputes will be resolved through arbitration.",
        "Assignment of this contract is not allowed without consent.",
        "Subcontracting requires prior approval.",
        "If any part of this agreement is invalid, the rest remains in effect.",
        "This document contains the entire agreement between the parties.",
        "Notices must be in writing and sent to the specified addresses.",
        "Amendments to this agreement must be in writing.",
        "Failure to enforce a clause does not waive future enforcement.",
        "Third-party rights are not granted under this agreement.",
        "Schedules attached to this agreement form part of it.",
        "Definitions provided in this section apply to the entire document.",
        "Execution of this agreement is binding on both parties.",
        "The representations made in this agreement are true and accurate.",
        "Guarantees provided are subject to the terms herein.",
        "Non-compete clauses apply for 2 years post-termination.",
        "Non-disclosure obligations remain for 5 years.",
        "Privacy policies must be adhered to at all times.",
        "Data protection measures must comply with GDPR.",
        "Insurance coverage must be maintained throughout the term.",
        "The company may audit compliance annually.",
        "Penalties for breach of contract are specified herein.",
        "Default by either party allows for termination.",
        "Performance metrics will be reviewed quarterly.",
        "Compliance with all relevant laws is required.",
        "Training for employees must be provided annually.",
        "Other terms may apply as mutually agreed.",
        "Notices must be acknowledged within 7 days.",
        "Terms of payment are net 30 days.",
        "Conditions of delivery are FOB origin.",
        "Liability is limited to the amount of the contract.",
        "Confidentiality agreements must be signed by all parties.",
        "Termination may occur with 30 days' notice."
    ],
    'type': [
        'price', 'delivery', 'payment', 'warranty', 'warranty',
        'governing law', 'force majeure', 'indemnity', 'intellectual property', 'dispute resolution',
        'assignment', 'subcontracting', 'severability', 'entire agreement', 'notice',
        'amendments', 'waiver', 'third-party rights', 'schedules', 'definitions',
        'execution', 'representations', 'guarantees', 'non-compete', 'non-disclosure',
        'privacy', 'data protection', 'insurance', 'audit', 'penalties',
        'default', 'performance', 'compliance', 'training', 'other',
        'notice', 'payment', 'delivery', 'liability', 'confidentiality',
        'termination'
    ],
    'category': [
        'main', 'sub', 'main', 'sub', 'main',
        'sub', 'main', 'sub', 'main', 'sub',
        'sub', 'sub', 'main', 'main', 'main',
        'main', 'sub', 'main', 'main', 'sub',
        'main', 'sub', 'main', 'sub', 'sub',
        'main', 'main', 'main', 'main', 'main',
        'sub', 'main', 'main', 'main', 'main',
        'main', 'sub', 'main', 'sub', 'main',
        'main'
    ]
}

# Verify lengths of arrays in data dictionary
assert len(data['text']) == len(data['type']) == len(data['category']), "All arrays in data must have the same length"

df = pd.DataFrame(data)

# Preprocess text
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

df['processed_text'] = df['text'].apply(preprocess_text)

# Vectorize text and split data
X = df['processed_text']
y = df[['type', 'category']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the multi-output model
model = make_pipeline(CountVectorizer(), MultiOutputClassifier(MultinomialNB()))
model.fit(X_train, y_train)

# Function to classify new text using NER entities
def classify_clause(text):
    processed_text = preprocess_text(text)
    doc = nlp(processed_text)
    
    # Initialize predictions based on the type and category classes
    type_pred = None
    category_pred = None
    
    # Extract named entities and classify based on predefined types and categories
    for ent in doc.ents:
        if ent.label_ == 'TYPE' and ent.text in data['type']:
            type_pred = ent.text
        elif ent.label_ == 'CATEGORY' and ent.text in data['category']:
            category_pred = ent.text
    
    # If entities are not found, fall back to the original classification method
    if not type_pred or not category_pred:
        type_pred, category_pred = model.predict([processed_text])[0]
    
    return type_pred, category_pred

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=BytesIO(uploaded_file.read()), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to identify headings and clauses
def segregate_headings_clauses(text):
    lines = text.split('\n')
    headings = []
    clauses = []
    current_heading = ""
    
    for line in lines:
        if re.match(r'^[A-Z\s]+$', line.strip()):
            if current_heading:
                headings.append(current_heading)
            current_heading = line.strip()
        else:
            clauses.append((current_heading, line.strip()))
    
    if current_heading:
        headings.append(current_heading)
    
    return headings, clauses

# Function to apply named entity recognition
def apply_ner(text):
    doc = nlp(text)
    for ent in doc.ents:
        # Annotate entities with predefined types and categories
        if ent.text in data['type']:
            ent.label_ = 'TYPE'
        elif ent.text in data['category']:
            ent.label_ = 'CATEGORY'
    return doc

