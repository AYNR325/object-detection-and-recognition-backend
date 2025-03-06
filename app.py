# import cv2
# import numpy as np
# import google.generativeai as genai
# from PIL import Image
# import io
# import base64
# import time
# import pytesseract
# from gtts import gTTS
# import pygame
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Allow cross-origin requests (for Java app)

# # Configure Google Gemini API
# GEMINI_API_KEY = "AIzaSyClwx3CyenZMAk5m9WqSw4_5ERuzXR7DCI"
# genai.configure(api_key=GEMINI_API_KEY)

# DEFAULT_LANGUAGE = "en"

# # Initialize Pygame for TTS
# pygame.mixer.init()

# def extract_text_from_image(image):
#     """Extract text from an image using Tesseract OCR."""
#     image_pil = Image.fromarray(image)
#     text = pytesseract.image_to_string(image_pil, lang='eng')
#     return text.strip() if text else "No text detected."

# def speak_text(text):
#     """Convert text to speech and play it."""
#     tts = gTTS(text=text, lang=DEFAULT_LANGUAGE, slow=False)
#     tts.save("output.mp3")
    
#     pygame.mixer.music.load("output.mp3")
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         time.sleep(0.1)

# def describe_image(image):
#     """Use Google's Gemini API to generate a detailed description of an image."""
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     image_pil = Image.fromarray(image)
    
#     # Convert image to Base64
#     img_byte_arr = io.BytesIO()
#     image_pil.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()
    
#     response = model.generate_content([
#         {"text": "Describe this image in 8-9 sentences in English."},
#         {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode('utf-8')}}
#     ], stream=False)
    
#     return response.text if response and hasattr(response, 'text') else "Could not generate description."

# @app.route('/process-image', methods=['POST'])
# def process_image():
#     """
#     Flask API Endpoint to process an image.
#     Expects a base64-encoded image and a mode ('description' or 'text').
#     """
#     data = request.get_json()
    
#     if 'image' not in data or 'mode' not in data:
#         return jsonify({"error": "Missing 'image' or 'mode' parameter"}), 400

#     try:
#         # Decode Base64 image
#         image_data = base64.b64decode(data['image'])
#         image_np = np.frombuffer(image_data, np.uint8)
#         image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

#         if data['mode'] == "description":
#             result = describe_image(image)
#         elif data['mode'] == "text":
#             result = extract_text_from_image(image)
#         else:
#             return jsonify({"error": "Invalid mode. Use 'description' or 'text'"}), 400

#         # Convert text to speech
#         speak_text(result)
        
#         return jsonify({"result": result})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import base64
import time
import pytesseract
from gtts import gTTS
import pygame
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (for Java app)

# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyClwx3CyenZMAk5m9WqSw4_5ERuzXR7DCI"
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_LANGUAGE = "en"

# Initialize Pygame for TTS
pygame.mixer.init()

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    image_pil = Image.fromarray(image)
    text = pytesseract.image_to_string(image_pil, lang='eng')
    return text.strip() if text else "No text detected."

def speak_text(text):
    """Convert text to speech and play it in a separate thread."""
    def play_audio():
        tts = gTTS(text=text, lang=DEFAULT_LANGUAGE, slow=False)
        tts.save("output.mp3")
        
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    # Run TTS in a separate thread to avoid blocking Flask
    threading.Thread(target=play_audio, daemon=True).start()

def describe_image(image):
    """Use Google's Gemini API to generate a detailed description of an image."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    image_pil = Image.fromarray(image)
    
    # Convert image to Base64
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    response = model.generate_content([
        {"text": "Describe this image in 8-9 sentences in English."},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode('utf-8')}}
    ], stream=False)

    return response.text if response and hasattr(response, 'text') else "Could not generate description."

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Flask API Endpoint to process an image.
    Expects a base64-encoded image and a mode ('description' or 'text').
    """
    data = request.get_json()
    
    if 'image' not in data or 'mode' not in data:
        return jsonify({"error": "Missing 'image' or 'mode' parameter"}), 400

    try:
        # Decode Base64 image
        image_data = base64.b64decode(data['image'])
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        if data['mode'] == "description":
            result = describe_image(image)
        elif data['mode'] == "text":
            result = extract_text_from_image(image)
        else:
            return jsonify({"error": "Invalid mode. Use 'description' or 'text'"}), 400

        # Convert text to speech asynchronously
        speak_text(result)
        
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
