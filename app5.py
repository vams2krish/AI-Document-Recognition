import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pytesseract
import spacy
import fitz  # PyMuPDF
from docx import Document

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("Document Upload and Text Extraction")

# Define colors for different categories
colors = {
    'long_sentences': (0, 0, 255),  # Red
    'referent_ambiguity': (255, 0, 0),  # Blue
    'coordination_ambiguity': (0, 255, 0),  # Green
    'speculative_modals': (255, 255, 0)  # Yellow
}

# Function to extract text from image
def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            st.error("Failed to load image for text extraction.")
            return ""
        text = pytesseract.image_to_string(image, config='--psm 6')
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# Function to get bounding box data for the text
def get_text_bounding_boxes(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            st.error("Failed to load image for bounding boxes.")
            return []

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        boxes = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                text = data['text'][i]
                boxes.append({'coords': (x, y, w, h), 'text': text})
        return boxes
    except Exception as e:
        st.error(f"Error getting bounding boxes: {e}")
        return []

# Function to highlight text areas with colors
def highlight_text_areas(image_path, text_areas, results):
    try:
        image = cv2.imread(image_path)
        if image is None:
            st.error("Failed to load image for highlighting.")
            return
        
        for area in text_areas:
            x, y, w, h = area['coords']
            text_type = area.get('type', 'default')
            color = colors.get(text_type, (255, 255, 255))  # Default to white
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Highlight rectangles
        
        # Save highlighted image
        highlighted_path = 'highlighted_image.png'
        cv2.imwrite(highlighted_path, image)
        
        # Show percentage counters
        total_texts = len(text_areas)
        counters = {key: len(value) for key, value in results.items()}
        st.write("Percentage Counters:")
        for category, count in counters.items():
            percentage = (count / total_texts) * 100 if total_texts > 0 else 0
            st.write(f"{category.replace('_', ' ').title()}: {count} ({percentage:.2f}%)")
        
        st.image(highlighted_path, caption="Highlighted Image with Annotations")
        
    except Exception as e:
        st.error(f"Error highlighting text areas: {e}")

# Function to analyze text with SpaCy
def analyze_text(text):
    doc = nlp(text)
    analysis_results = {
        "long_sentences": [],
        "speculative_modals": [],
        "coordination_ambiguity": [],
        "referent_ambiguity": []
    }

    # Detect long sentences
    for sent in doc.sents:
        if len(sent) > 100:
            analysis_results["long_sentences"].append(sent.text)

    # Detect speculative modals
    speculative_words = {"may", "might", "could", "perhaps", "possibly"}
    for token in doc:
        if token.text.lower() in speculative_words:
            analysis_results["speculative_modals"].append(token.text)

    # Detect coordination ambiguity
    for token in doc:
        if token.dep_ == "conj":
            analysis_results["coordination_ambiguity"].append(token.text)

    # Detect referent ambiguity
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"] and not token.ent_type_:
            analysis_results["referent_ambiguity"].append(token.text)

    return analysis_results

# Process the uploaded file
def process_file(file):
    file_type = file.type.split('/')[1]
    
    # Save the file to disk
    file_path = f'uploaded_file.{file_type}'
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    # Extract text based on file type
    if file_type in ["png", "jpg", "jpeg"]:
        # Image processing
        st.image(file_path, caption="Uploaded Image")
        text = extract_text_from_image(file_path)
        st.write("Extracted Text:")
        st.write(text)
        text_areas = get_text_bounding_boxes(file_path)
        analysis = analyze_text(text)
        st.write("Text Analysis Results:")
        st.json(analysis)
        if text_areas:
            highlight_text_areas(file_path, text_areas, analysis)
        else:
            st.write("No text areas detected for highlighting.")
    elif file_type == "pdf":
        # PDF processing
        text = extract_text_from_pdf(file_path)
        st.write("Extracted Text from PDF:")
        st.write(text)
        analysis = analyze_text(text)
        st.write("Text Analysis Results:")
        st.json(analysis)
    elif file_type == "docx":
        # DOCX processing
        text = extract_text_from_docx(file_path)
        st.write("Extracted Text from DOCX:")
        st.write(text)
        analysis = analyze_text(text)
        st.write("Text Analysis Results:")
        st.json(analysis)
    else:
        st.error("Unsupported file type.")

# Streamlit UI for file upload
uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "pdf", "docx"])

if uploaded_file:
    process_file(uploaded_file)
else:
    st.write("No file uploaded. Please upload a file to continue.")
