from flask import Flask, jsonify, request
from flask_cors import CORS
import assemblyai as aai
# from summarydepend import *;
from translate import Translator

app = Flask(__name__)
CORS(app)
aai.settings.api_key = "44527a6a3d1e42c38a3d7dc6d7dc8cf3"
API = "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"

@app.route('/', methods=['GET'])
def getrequest():
    return jsonify({'message': "This server doesn't accept GET request" }),500


@app.route('/audio-transcribe', methods=['POST'])
def audio_transcribe():
    try:
        print("entered")
        audio_file = request.files['audioFile'] 
        print(audio_file)
        # Save the file temporarily if needed
        audio_file_path = 'temp_audio.wav'
        audio_file.save(audio_file_path)
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file_path)
        # You can do further processing with the transcript if needed
        print(transcript.text)
        return jsonify({'data': transcript.text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def conversion(input_text, target_language):
    translator = Translator(to_lang=target_language)
    result = translator.translate(input_text)
    return result

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        print("entered")
        data = request.get_json()
        print(data)

        if 'text' not in data or 'target_language' not in data:
            return jsonify({'error': 'Invalid input. Please provide both "text" and "target_lang" parameters.'}), 400

        text = data['text']
        target_lang = data['target_language']

        translated_text = conversion(text, target_lang)
        print(translate_text)
        return jsonify({'translated_text': translated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# @app.route('/AI-API/text-summarization', methods=['POST'])
# def summarize_document():
#     try:
#         results = []
#         _input = request.form.getlist('input_content')
#         if  request.form['document_type'] == "text":
#             document_type = "large"
#             if len(_input[0]) > 4000:
#                 text_chunks = divide_text(_input[0])
#                 result=""
#                 for chunk in text_chunks:
#                     result += summarize_document_with_retry(chunk, document_type,API)
#                 results.append(result)
#             else:
#                 result = summarize_document_with_retry(_input[0], document_type,API)
#                 results.append(result)
#         else:
#             _files = request.files.getlist('pdf')
#             _files += request.files.getlist('docx')
#             _types = request.form.getlist('document_type')
            
#             for pdf_file, document_type in zip(_files, _types):
#                 extracted_text = extracted_data(pdf_file)
#                 if len(extracted_text) > 4000:
#                     text_chunks = divide_text(extracted_text)
#                     result = ""
#                     for chunk in text_chunks:
#                         result += summarize_document_with_retry(chunk, document_type,API)
#                     results.append(result)
#                 else:
#                     result = summarize_document_with_retry(extracted_text, document_type,API)
#                     results.append(result)
#         cleaned_result = []
#         for i in results:
#             x = []
#             x.append(clean_text(i))
#             cleaned_result.append(x)
#         return jsonify({'results': cleaned_result})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    



# from flask import Flask, request, jsonify
from google.cloud import vision
import os
import math
from collections import Counter
import re

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'nth-bucksaw-411711-39dbab0ccf11.json'
# app = Flask(_name_)

def detect_text(image_blob):
    """Detects text in the blob."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_blob)

    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    ocr_text = []
    for text in texts:
        ocr_text.append(f"\r\n{text.description}")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return ocr_text

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request.'}), 400

        image_blob = request.files['image'].read()

        ocr_result = detect_text(image_blob)

        return jsonify({'ocr_text': ocr_result[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if _name_ == '_main_':
#     app.run(debug=True, port=6000)

if __name__ == '__main__':
    app.run(debug=True)






# from translate import Translator

# def conversion(input_text, target_language):
#   translator = Translator(to_lang = target_language)
#   result = translator.translate(input_text)
#   return result

# if __name__ == '__main__':
#   text = input("Enter the text: ")
#   target_lang = input("Enter your target language: ")
#   translated_text = conversion(text, target_lang)
#   print(translated_text)