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

# Streamlit UI for image upload
st.title("Document Image Upload and Text Extraction")

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
        
        colors = {
            'long_sentence': (0, 0, 255),  # Red
            'referent_ambiguity': (255, 0, 0),  # Blue
            'coordination_ambiguity': (0, 255, 0),  # Green
            'speculative_modal': (255, 255, 0)  # Yellow
        }
        
        for area in text_areas:
            x, y, w, h = area['coords']
            text_type = area.get('type', 'default')
            color = colors.get(text_type, (255, 255, 255))  # Default to white
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Highlight rectangles
        
        # Save highlighted image
        cv2.imwrite('highlighted_image.png', image)
        
        # Show percentage counters
        total_texts = len(text_areas)
        counters = {key: len(value) for key, value in results.items()}
        st.write("Percentage Counters:")
        for category, count in counters.items():
            percentage = (count / total_texts) * 100 if total_texts > 0 else 0
            st.write(f"{category.replace('_', ' ').title()}: {count} ({percentage:.2f}%)")
        
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

# Process the uploaded image
def process_image(image_file):
    # Convert the image to OpenCV format
    img = Image.open(image_file)
    img_array = np.array(img)

    # Save the image to disk for further processing
    cv2.imwrite('uploaded_image.png', img_array)
    st.image('uploaded_image.png', caption="Uploaded Image")

    # Extract text from the image
    text = extract_text_from_image('uploaded_image.png')
    st.write("Extracted Text:")
    st.write(text)

    # Perform text analysis and display results
    analysis = analyze_text(text)
    st.write("Text Analysis Results:")
    st.json(analysis)

    # Get bounding boxes for text
    text_areas = get_text_bounding_boxes('uploaded_image.png')

    # Annotate text areas based on the analysis
    annotated_areas = []
    for area in text_areas:
        text = area['text']
        if text in analysis["long_sentences"]:
            area['type'] = 'long_sentence'
        elif text in analysis["referent_ambiguity"]:
            area['type'] = 'referent_ambiguity'
        elif text in analysis["coordination_ambiguity"]:
            area['type'] = 'coordination_ambiguity'
        elif text in analysis["speculative_modals"]:
            area['type'] = 'speculative_modal'
        else:
            area['type'] = 'default'
        annotated_areas.append(area)

    # Highlight text areas in the image
    if annotated_areas:
        highlight_text_areas('uploaded_image.png', annotated_areas, analysis)
        st.image('highlighted_image.png', caption="Highlighted Image with Annotations")
    else:
        st.write("No text areas detected for highlighting.")

# Streamlit UI for image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    process_image(uploaded_file)
else:
    st.write("No image uploaded. Please upload an image to continue.")
