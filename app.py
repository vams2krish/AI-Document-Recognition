import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pytesseract
import spacy

# Provide the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI for image capture
st.title("Document Image Capture and Text Extraction")

# Function to extract text from image
def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        # Ensure image is read correctly
        if image is None:
            st.error("Failed to load image for text extraction.")
            return ""
        
        # Perform OCR
        text = pytesseract.image_to_string(image, config='--psm 6')  # Use page segmentation mode 6
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
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
            if int(data['conf'][i]) > 60:  # Confidence threshold
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                boxes.append({'coords': (x, y, w, h)})
        return boxes
    except Exception as e:
        st.error(f"Error getting bounding boxes: {e}")
        return []

# Function to highlight text areas
def highlight_text_areas(image_path, text_areas):
    try:
        image = cv2.imread(image_path)
        if image is None:
            st.error("Failed to load image for highlighting.")
            return
        
        for area in text_areas:
            x, y, w, h = area['coords']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangles
        cv2.imwrite('highlighted_image.png', image)
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
        if len(sent) > 100:  # Define a threshold for "long"
            analysis_results["long_sentences"].append(sent.text)

    # Detect speculative modals
    speculative_words = {"may", "might", "could", "perhaps", "possibly"}
    for token in doc:
        if token.text.lower() in speculative_words:
            analysis_results["speculative_modals"].append(token.text)

    # Detect coordination ambiguity and referent ambiguity
    for token in doc:
        if token.dep_ == "conj":
            analysis_results["coordination_ambiguity"].append(token.text)
        if token.dep_ == "nsubj" or token.dep_ == "dobj":
            if token.ent_type_ == "":
                analysis_results["referent_ambiguity"].append(token.text)

    return analysis_results

# Capture and process the image
def process_image(image_file):
    # Convert the image to OpenCV format
    img = Image.open(image_file)
    img_array = np.array(img)

    # Save the image to disk for further processing
    cv2.imwrite('captured_image.png', img_array)
    st.image('captured_image.png', caption="Captured Image")

    # Extract text from the image
    text = extract_text_from_image('captured_image.png')
    st.write("Extracted Text:")
    st.write(text)

    # Perform text analysis and display results
    analysis = analyze_text(text)
    st.write("Text Analysis Results:")
    st.json(analysis)

    # Get bounding boxes for text and highlight them
    text_areas = get_text_bounding_boxes('captured_image.png')
    if text_areas:
        highlight_text_areas('captured_image.png', text_areas)
        st.image('highlighted_image.png', caption="Highlighted Image with Text")
    else:
        st.write("No text areas detected for highlighting.")

# Streamlit UI for image capture
image_file = st.camera_input("Take a picture")

if image_file:
    process_image(image_file)
else:
    st.write("No image captured. Please take a picture to continue.")
