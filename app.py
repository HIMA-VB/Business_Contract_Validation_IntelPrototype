import streamlit as st
import pandas as pd
import tempfile
import torch
import torch.nn.functional as F
from compare_highlight import *
from transformers import AutoTokenizer, AutoModel
from text_classifier import extract_text_from_pdf, classify_clause, segregate_headings_clauses
from summarizer import compare_pdf

# Initialize the pre-trained model and tokenizer
def load_model_and_tokenizer():
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

# Function to compare sentence similarity
def compare_sentence_similarity(sentences1, sentences2, model, tokenizer):
    # Tokenize sentences
    inputs = tokenizer(sentences1 + sentences2, return_tensors='pt', max_length=512, truncation=True, padding=True)

    # Obtain sentence embeddings
    with torch.no_grad():
        model_output = model(**inputs)

    # Mean pooling to get a single vector representation for each sentence
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1)

    # Calculate cosine similarities
    similarities = []
    for i in range(len(sentences1)):
        sim = F.cosine_similarity(sentence_embeddings[i], sentence_embeddings[len(sentences1):], dim=1)
        similarities.append(sim)

    return similarities

def main_streamlit():
    st.title("Business Contract Validation")

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'highlighted_pdf1' not in st.session_state:
        st.session_state.highlighted_pdf1 = None
    if 'highlighted_pdf2' not in st.session_state:
        st.session_state.highlighted_pdf2 = None
    if 'continue_search' not in st.session_state:
        st.session_state.continue_search = False
    if 'search_conditions' not in st.session_state:
        st.session_state.search_conditions = []

    # File upload
    uploaded_file1 = st.file_uploader("Upload PDF Document 1", type="pdf")
    uploaded_file2 = st.file_uploader("Upload PDF Document 2", type="pdf")

    if uploaded_file1 and uploaded_file2 and not st.session_state.processed:
        st.write("Parsing PDF documents...")

        # Extract and process the first PDF using compare_highlight.py
        if uploaded_file1.name:
            st.write(f"Processing PDF: {uploaded_file1.name}")
            pdf_text1 = extract_text_from_pdf(uploaded_file1)
            headings1, contents1 = separate_headings_and_content(pdf_text1)
            positions1 = extract_text_and_positions_from_pdf(uploaded_file1.name)  # Check if .name is correct

            if headings1 and contents1:
                data1 = {'Heading': headings1, 'Content': contents1}
                df1 = pd.DataFrame(data1)
                st.subheader("First PDF")
                st.write(df1)
                st.write("---")  # Add a separator

                st.write("Text Classification Results for First PDF")
                for content in contents1:
                    type_pred, category_pred = classify_clause(content)
                    st.write(f"Content: {content}")
                    st.write(f"Type Prediction: {type_pred}")
                    st.write(f"Category Prediction: {category_pred}")
                    st.write("-" * 50)

        # Extract and process the second PDF using compare_highlight.py
        if uploaded_file2.name:
            st.write(f"Processing PDF: {uploaded_file2.name}")
            pdf_text2 = extract_text_from_pdf(uploaded_file2)
            headings2, contents2 = separate_headings_and_content(pdf_text2)
            positions2 = extract_text_and_positions_from_pdf(uploaded_file2.name)  # Check if .name is correct

            if headings2 and contents2:
                data2 = {'Heading': headings2, 'Content': contents2}
                df2 = pd.DataFrame(data2)
                st.subheader("Second PDF")
                st.write(df2)
                st.write("---")  # Add a separator

                st.write("Text Classification Results for Second PDF")
                for content in contents2:
                    type_pred, category_pred = classify_clause(content)
                    st.write(f"Content: {content}")
                    st.write(f"Type Prediction: {type_pred}")
                    st.write(f"Category Prediction: {category_pred}")
                    st.write("-" * 50)

        # Compare headings and detect differences using compare_highlight.py
        heading_deviations = compare_headings(headings1, headings2) if headings1 and headings2 else []

        if heading_deviations:
            st.subheader("Heading Deviations")
            for deviation in heading_deviations:
                st.write(f"Deviation: {deviation['deviation']}")
                st.write(f"Heading: {deviation['heading']}")
                st.write("-" * 50)
        else:
            st.write("No heading deviations found.")

        # Load pre-trained model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        # Compare section names and perform semantic analysis using compare_highlight.py
        common_section_names = set(headings1).intersection(set(headings2)) if headings1 and headings2 else []

        all_deviations = []

        if common_section_names:
            st.subheader("Semantic Deviations in Section Contents")

            for section_name in common_section_names:
                idx1 = headings1.index(section_name) if section_name in headings1 else None
                idx2 = headings2.index(section_name) if section_name in headings2 else None

                if idx1 is not None and idx2 is not None:
                    content1 = contents1[idx1].split('\n') if contents1 else []
                    content2 = contents2[idx2].split('\n') if contents2 else []

                    # Compare semantics of sentences using compare_highlight.py
                    deviations = compare_semantics(content1, content2, model, tokenizer)

                    if deviations:
                        st.write(f"Section Name: {section_name}")
                        for deviation in deviations:
                            st.write(f"PDF 1 Difference: {deviation['pdf1_diff']}")
                            st.write(f"PDF 2 Difference: {deviation['pdf2_diff']}")
                            st.write(f"Similarity Score: {deviation['similarity_score']:.2f}")
                            st.write("-" * 50)

                        all_deviations.extend(deviations)

                    else:
                        st.write(f"No semantic deviations found for section: {section_name}")

                else:
                    st.write(f"Section '{section_name}' not found in both PDFs.")

        else:
            st.write("No common section names found between the two PDFs.")

        all_deviations.extend(heading_deviations)

        if all_deviations:
            # Highlight deviations in both PDFs
            highlighted_pdf1 = highlight_deviations_in_pdf(uploaded_file1.name, all_deviations, positions1) if uploaded_file1 else None
            highlighted_pdf2 = highlight_deviations_in_pdf(uploaded_file2.name, all_deviations, positions2) if uploaded_file2 else None

            # Save highlighted PDFs in session state
            st.session_state.highlighted_pdf1 = highlighted_pdf1
            st.session_state.highlighted_pdf2 = highlighted_pdf2

        # Mark as processed to avoid reprocessing
        st.session_state.processed = True

    # Display download buttons after processing
    if st.session_state.processed and st.session_state.highlighted_pdf1 and st.session_state.highlighted_pdf2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file1:
            tmp_file1.close()
            st.session_state.highlighted_pdf1.save(tmp_file1.name)
            st.subheader("Download Highlighted First PDF")
            with open(tmp_file1.name, "rb") as f:
                st.download_button(label="Download PDF", data=f, file_name="highlighted_pdf1.pdf")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file2:
            tmp_file2.close()
            st.session_state.highlighted_pdf2.save(tmp_file2.name)
            st.subheader("Download Highlighted Second PDF")
            with open(tmp_file2.name, "rb") as f:
                st.download_button(label="Download PDF", data=f, file_name="highlighted_pdf2.pdf")

        # Ask user if they want to continue with rule-based search
        st.subheader("Continue with Rule-based Search?")
        continue_search = st.checkbox("Yes")

        if continue_search:
            st.session_state.continue_search = True

    # Rule-based search if user chooses to continue
    if st.session_state.continue_search:
        st.subheader("Rule-based Search")

        # Prompt user for conditions (sentences)
        condition = st.text_input("Enter a condition (sentence):")

        if st.button("Add Condition") and condition.strip():
            st.session_state.search_conditions.append(condition.strip())

        # Display current conditions
        st.subheader("Current Conditions")
        st.write(st.session_state.search_conditions)

        # Ask user to upload PDFs again
        st.subheader("Upload PDFs Again for Rule-based Search")
        re_uploaded_file1 = st.file_uploader("Upload PDF Document 1 for Rule-based Search", type="pdf")
        re_uploaded_file2 = st.file_uploader("Upload PDF Document 2 for Rule-based Search", type="pdf")

        if re_uploaded_file1 and re_uploaded_file2:
            # Save the uploaded files
            with open("re_document1.pdf", "wb") as f1, open("re_document2.pdf", "wb") as f2:
                f1.write(re_uploaded_file1.getbuffer())
                f2.write(re_uploaded_file2.getbuffer())

            # Process each PDF
            model, tokenizer = load_model_and_tokenizer()

            for pdf_num, re_uploaded_file in enumerate([re_uploaded_file1, re_uploaded_file2], start=1):
                st.write(f"Searching in Re-uploaded PDF {pdf_num}...")

                # Extract text from re-uploaded PDF
                text = extract_text_from_pdf(re_uploaded_file)

                # Split text into sentences
                sentences = text.split('\n')

                # Compare each sentence with conditions
                for condition in st.session_state.search_conditions:
                    similarities = compare_sentence_similarity([condition], sentences, model, tokenizer)
                    for i, similarity in enumerate(similarities[0]):
                        if similarity.item() >= 0.80:
                            st.write(f"PDF {pdf_num} - Match found:")
                            st.write(f"Condition: {condition}")
                            st.write(f"Sentence: {sentences[i]}")
                            st.write(f"Similarity Score: {similarity.item():.2f}")
                            st.write("-" * 50)
    if uploaded_file1 and uploaded_file2:
        with open("uploaded_file1_summary.pdf", "wb") as f1, open("uploaded_file2_summary.pdf", "wb") as f2:
            f1.write(uploaded_file1.getbuffer())
            f2.write(uploaded_file2.getbuffer())

            summary1, summary2 = compare_pdf("uploaded_file1_summary.pdf", "uploaded_file2_summary.pdf")


        st.write("Summary of PDF 1:")
        st.write(summary1)

        st.write("Summary of PDF 2:")
        st.write(summary2)


if __name__ == "__main__":
    main_streamlit()
