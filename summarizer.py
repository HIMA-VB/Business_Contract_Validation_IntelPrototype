import fitz  # PyMuPDF
from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization")

def compare_pdf(uploaded_file1, uploaded_file2):
    summary1 = ""
    summary2 = ""

    # Open the PDF files
    with open(uploaded_file1, 'rb') as file1, open(uploaded_file2, 'rb') as file2:
        # Open the PDFs with PyMuPDF
        doc1 = fitz.open(stream=file1.read(), filetype="pdf")
        doc2 = fitz.open(stream=file2.read(), filetype="pdf")

        # Get the number of pages in each PDF
        num_pages = min(len(doc1), len(doc2))

        # Iterate through the pages and summarize them
        for page_num in range(num_pages):
            # Get the text of each page
            page1_text = doc1[page_num].get_text()
            page2_text = doc2[page_num].get_text()

            # Summarize the text of the pages
            summary1 += summarize_text(page1_text)
            summary2 += summarize_text(page2_text)

    return summary1, summary2

def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def highlight_differences(page1, page2):
    # Get the text content of each page
    text1 = page1.get_text()
    text2 = page2.get_text()

    # Split the text into words
    words1 = set(text1.split())
    words2 = set(text2.split())

    # Find the differences between the words in both pages
    differences1 = words1 - words2
    differences2 = words2 - words1

    # Highlight the differences on the page
    for diff_word in differences1:
        highlight_word(page1, diff_word, color=(1, 0, 0))  # Red color for differences in page 1
    for diff_word in differences2:
        highlight_word(page2, diff_word, color=(0, 1, 0))  # Green color for differences in page 2

def highlight_word(page, word, color):
    # Iterate through the words in the page and find the position of the target word
    for b in page.search_for(word):
        rect = fitz.Rect(b[:4])  # Get the coordinates of the word
        highlight = page.add_highlight_annot(rect)
        highlight.update()  # Highlight the word
