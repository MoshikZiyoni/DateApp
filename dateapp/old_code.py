import cv2
import numpy as np
# import pytesseract
import re

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\moshi\DateApp\Tesseract-OCR\tesseract.exe'

import os
# # os.environ['TESSDATA_PREFIX'] = '/usr/local/share/tessdata'



def detect_text_bubbles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return []
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # Adjust the threshold value as needed

    # Dilate the thresholded image to merge text into blobs
    kernel = np.ones((10,10), np.uint8)  # Adjust the kernel size as needed
    dilation = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw contours
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtering the contours based on area and aspect ratio of the text bubble
        if 1000 < w*h < 50000:  # Adjust the area range as needed
            boxes.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the result with detected squares only
    # cv2.imshow('Detected Text Bubbles', image)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()
    boxes.sort(key=lambda b: b[1])
    return boxes

def classify_and_crop_text_areas(image, boxes):
    mid_x = image.shape[1] // 2
    cropped_images = []
    for index, (x, y, w, h) in enumerate(boxes):
        label = "male" if x + w // 2 < mid_x else "female"
        cropped = image[y:y+h, x:x+w]
        # Include index to maintain the order
        cropped_images.append((cropped, label, index))
    return cropped_images

# def extract_and_print_text(cropped_images):
#     texts = []
#     for cropped, label, index in cropped_images:
#         # Extract text from the cropped image using Tesseract OCR
#         text = pytesseract.image_to_string(cropped, lang='heb')
#         # Clean up the text from OCR
#         text = re.sub(r'\s+', ' ', text).strip()
#         # Reverse the text for correct Hebrew display
#         text = text[::-1]  # This reverses the text
#         texts.append((text, label, index))
#     return texts
def extract_and_print_text(cropped_images):
    texts = []
    for cropped, label, index in cropped_images:
        # Extract text from the cropped image using Tesseract OCR
        text = pytesseract.image_to_string(cropped, lang='heb')
        # print(text)
        # Clean up the text from OCR
        text = re.sub(r'\s+', ' ', text).strip()
        # Reverse the text for correct Hebrew display
        text = text[::-1]  # This reverses the text
        # Remove timestamp-like strings
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)
         # Remove irrelevant parts
        text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
        text = re.sub(r'\b\d+(?=[^\W\d_])', '', text)  # Remove numbers followed by non-numeric characters

        text = re.sub(r'ציהנפל', '', text)  # Remove specific string
        text = re.sub(r'צ"הנפל', '', text)  # Remove specific string
        text = re.sub(r'ציחנפל', '', text)  # Remove specific string
        text = re.sub(r'צ-הנפל', '', text)  # Remove specific string
        text = re.sub(r'ציהחא', '', text)  # Remove specific string
        text = re.sub(r'צ"הנפכ', '', text)  # Remove specific string
        text = re.sub(r'צ"החהא', '', text)  # Remove specific string
        text = re.sub(r'ציחחא', '', text)  # Remove specific string
        text = re.sub(r'ציההא', '', text)  # Remove specific string
        text = re.sub(r"'צהחא", '', text)  # Remove specific string
        text = re.sub(r'צ ₪', '', text)  # Remove specific string
        text = re.sub(r'-> ₪ ןשי', '', text)  # Remove specific string
        text = re.sub(r'-', '', text)  # Remove specific string
        text = re.sub(r'/', '', text)  # Remove specific string
        # text = re.sub(r')', '', text)  # Remove specific string

        # Append only if the text is not empty
        if text:
            gender = "MALE" if label == 'male' else "FEMALE"
            formatted_text = f"{text}-{gender}"
            texts.append(formatted_text)
    return texts




def save_and_display_results(cropped_images):
    
    formatted_texts = extract_and_print_text(cropped_images)
    for text in formatted_texts:
        print(text)
    # texts = extract_and_print_text(cropped_images)
    # print(texts)
    # for i, (text, label, index) in enumerate(texts):
    #     if is_text_letters_only(text):
    #         cropped_path = f"cropped_{index}_{label}.png"
    #         cv2.imwrite(cropped_path, cropped_images[i][0])  # Save the cropped image
    #         print(f"Saved: {cropped_path}")
    #         # Print the text extracted from the cropped image
    #         print(f"Text (Index: {index}, Label: {label}): {text}")
    #     else:
    #         # print(f"Skipped non-letter text (Index: {index}, Label: {label})")
    #         print('')



import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\moshi\DateApp\Tesseract-OCR2\tesseract.exe"
# 
image_path = r"C:\Users\moshi\DateApp\aaaa.png"  # Change to your actual image path
# image = Image.open(image_path)
image = cv2.imread(image_path)

boxes = detect_text_bubbles(image_path)
cropped_images = classify_and_crop_text_areas(image, boxes)
save_and_display_results(cropped_images)

# ocr_result=pytesseract.image_to_string(image,lang='heb')
# print(ocr_result)