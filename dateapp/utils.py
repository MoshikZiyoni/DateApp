import cv2
import numpy as np
import re
import os
import pytesseract
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import uuid
import requests
from io import BytesIO
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

os.environ['TESSDATA_PREFIX'] = '/usr/local/share/tessdata'
# Get the directory path of the current Python script
# current_dir = os.path.dirname(os.path.abspath(__file__))


# pytesseract.pytesseract.tesseract_cmd = os.path.join(current_dir, 'Tesseract-OCR2', 'tesseract.exe')
# # Construct the path to the pytesseract executable
# pytesseract.pytesseract.tesseract_cmd=f"{current_dir}\\Tesseract-OCR2\\tesseract.exe"
# pytesseract_path = f"{current_dir}\\Tesseract-OCR2\\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\moshi\DateApp\dateapp\Tesseract-OCR2\tesseract.exe"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh

def detect_text_bubbles(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return []

    preprocessed = preprocess_image(image)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(preprocessed, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 1000 < w*h < 50000:
            boxes.append((x, y, w, h))
    
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


def extract_and_print_text(cropped_images):
    texts = []
    for cropped, label, index in cropped_images:
        # Extract text from the cropped image using Tesseract OCR
        text = pytesseract.image_to_string(cropped, lang='heb')
        
        # Clean up the text from OCR
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[::-1]  # Reverse the text for correct Hebrew display
        
        # Remove timestamp-like strings and irrelevant parts
        text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
        text = re.sub(r'\b\d+\b|\b\d+(?=[^\W\d_])', '', text)
        
        # Remove specific strings
        text = re.sub(r'(ציהנפל|צ"הנפל|ציחנפל|צ-הנפל|ציהחא|צ"הנפכ|צ"החהא|ציחחא|ציההא|\'צהחא|צ ₪|-> ₪ ןשי|-|צ"החא|/)', '', text)
        
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
    

        

@csrf_exempt
def show_text_from_image(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            print(current_dir)
            print()
            print()
            print()
    #         # Read image file from request
            image_file = request.FILES['image']

            # Generate unique filename
            filename = str(uuid.uuid4()) + '.png'
            
            # Save the image temporarily
            with open(filename, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
        elif 'image_url' in request.POST:
            # Extract image URL from request
            image_url = request.POST['image_url']
            
            # Download the image from the URL
            response = requests.get(image_url)
            
            if response.status_code == 200:
                # Read the downloaded image
                image_data = BytesIO(response.content)
                image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)

                # Generate unique filename
                filename = str(uuid.uuid4()) + '.png'
                
                # Save the image temporarily
                cv2.imwrite(filename, image)
            else:
                return JsonResponse({'error': 'Failed to download image from URL'}, status=400)
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)

        # Get the full file path
        file_path = os.path.abspath(filename)

        # Read the saved image
        image = cv2.imread(file_path)

        boxes = detect_text_bubbles(file_path)
        cropped_images = classify_and_crop_text_areas(image, boxes)
        texts = extract_and_print_text(cropped_images)

        # Delete the temporary image file
        os.remove(filename)

        return JsonResponse({'texts': texts})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
