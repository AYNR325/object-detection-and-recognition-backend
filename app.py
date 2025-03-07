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

#  this code sends a text and audio file (mp3) in base64 format as JSON response to the client  and then at client side the audio file is decoded and played.
# import cv2
# import numpy as np
# import google.generativeai as genai
# from PIL import Image
# import io
# import base64
# import pytesseract
# from gtts import gTTS
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  

# # Configure Google Gemini API
# GEMINI_API_KEY = "AIzaSyClwx3CyenZMAk5m9WqSw4_5ERuzXR7DCI"
# genai.configure(api_key=GEMINI_API_KEY)

# DEFAULT_LANGUAGE = "en"

# def extract_text_from_image(image):
#     """Extract text from an image using Tesseract OCR."""
#     image_pil = Image.fromarray(image)
#     text = pytesseract.image_to_string(image_pil, lang='eng')
#     return text.strip() if text else "No text detected."

# def describe_image(image):
#     """Use Google's Gemini API to generate a detailed description of an image."""
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     image_pil = Image.fromarray(image)
    
#     img_byte_arr = io.BytesIO()
#     image_pil.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()
    
#     response = model.generate_content([
#         {"text": "Describe this image in 8-9 sentences in English."},
#         {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode('utf-8')} }
#     ], stream=False)

#     return response.text if response and hasattr(response, 'text') else "Could not generate description."

# def generate_speech(text):
#     """Convert text to speech and return as a byte stream."""
#     tts = gTTS(text=text, lang=DEFAULT_LANGUAGE, slow=False)
#     audio_io = io.BytesIO()
#     tts.write_to_fp(audio_io)  
#     audio_io.seek(0)
#     return audio_io

# @app.route('/process-image', methods=['POST'])
# def process_image():
#     """Processes the image, returns JSON text first, then allows fetching MP3."""
#     data = request.get_json()
    
#     if 'image' not in data or 'mode' not in data:
#         return jsonify({"error": "Missing 'image' or 'mode' parameter"}), 400

#     try:
#         # Decode Base64 image
#         image_data = base64.b64decode(data['image'])
#         image_np = np.frombuffer(image_data, np.uint8)
#         image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         if data['mode'] == "description":
#             result_text = describe_image(image)
#         elif data['mode'] == "text":
#             result_text = extract_text_from_image(image)
#         else:
#             return jsonify({"error": "Invalid mode. Use 'description' or 'text'"}), 400

#         # Generate speech
#         audio_io = generate_speech(result_text)

#         # Store audio in memory
#         audio_data = audio_io.getvalue()

#         # ✅ Send JSON response with Base64 audio
#         response = {
#             "text": result_text,
#             "audio": base64.b64encode(audio_data).decode('utf-8')  # Convert audio to Base64
#         }
#         return jsonify(response), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# this code send text and audio file (mp3) directly as a response to the request made by the client
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import base64
import pytesseract
from gtts import gTTS
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyClwx3CyenZMAk5m9WqSw4_5ERuzXR7DCI"
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_LANGUAGE = "en"

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    image_pil = Image.fromarray(image)
    text = pytesseract.image_to_string(image_pil, lang='eng')
    return text.strip() if text else "No text detected."

def describe_image(image):
    """Use Google's Gemini API to generate a detailed description of an image."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    image_pil = Image.fromarray(image)
    
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    response = model.generate_content([
        {"text": "Describe this image in 8-9 sentences in English."},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode('utf-8')} }
    ], stream=False)

    return response.text if response and hasattr(response, 'text') else "Could not generate description."

def generate_speech(text):
    """Convert text to speech and return as an audio file (MP3)."""
    tts = gTTS(text=text, lang=DEFAULT_LANGUAGE, slow=False)
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)  
    audio_io.seek(0)
    return audio_io

@app.route('/process-image', methods=['POST'])
def process_image():
    """Processes the image and returns both text and MP3 in a single response."""
    data = request.get_json()
    
    if 'image' not in data or 'mode' not in data:
        return jsonify({"error": "Missing 'image' or 'mode' parameter"}), 400

    try:
        # Decode Base64 image
        image_data = base64.b64decode(data['image'])
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if data['mode'] == "description":
            result_text = describe_image(image)
        elif data['mode'] == "text":
            result_text = extract_text_from_image(image)
        else:
            return jsonify({"error": "Invalid mode. Use 'description' or 'text'"}), 400

        # Generate speech (MP3)
        audio_io = generate_speech(result_text)

        # ✅ Send multipart response (JSON + MP3)
        boundary = "----MultipartBoundary"
        response_body = f"--{boundary}\r\n"
        response_body += "Content-Type: application/json\r\n\r\n"
        response_body += jsonify({"text": result_text}).data.decode("utf-8")
        response_body += f"\r\n--{boundary}\r\n"
        response_body += "Content-Type: audio/mpeg\r\n\r\n"

        def generate():
            """Generate response chunks."""
            yield response_body.encode("utf-8")
            yield audio_io.read()
            yield f"\r\n--{boundary}--\r\n".encode("utf-8")

        return Response(generate(), mimetype=f"multipart/mixed; boundary={boundary}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
