from flask import Flask, jsonify, request,send_file
from flask_cors import CORS
import assemblyai as aai
import pdfplumber
from gtts import gTTS
# from summarydepend import *;
from translate import Translator
import fitz

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
        
        # print("entered")
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



def extract_text_from_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]
            text += page.extract_text()
    return text


def text_to_speech(text, language, filename):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(filename)


@app.route('/text-to-speech', methods=['POST'])
def voiceGeneration():
    try:
        print("entered to module")
        input_type = request.form['input_type']
        print(input_type)
        if input_type == 'text':
            inp_text = request.form['input_text']
            print(inp_text)
            text_to_speech(inp_text, 'en', 'testing-speech.wav')
        else:
            input_file = request.files['input_document']
            extracted_text = extract_text_from_pdfplumber(input_file)
            print(extracted_text)
            text_to_speech(extracted_text, 'en', 'testing-speech.wav')

        return send_file('testing-speech.wav', as_attachment=True,mimetype='audio/mpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def conversion(input_text, target_language):
    translator = Translator(to_lang=target_language)
    result = translator.translate(input_text)
    return result

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        print("entered to lang-translation")
        input_type=request.form['input_type']
        target_lang=request.form['input_language']
      
        # print(request.form['text'])
        print(target_lang,"printing target lang")
       
        if (input_type == 'text'):
            print("sucesss")
            text=request.form['input_text']
           

        else:
            input_file=request.files['input_document']
            text=extract_text_from_pdfplumber(input_file)
            
        print(text)    
        translated_text = conversion(text, target_lang)

        print(translated_text)
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
# import fitz
import os
import math
from collections import Counter
import re
# packages

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'nth-bucksaw-411711-9272bdfb012d.json'
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
        if 'image' in request.files:
            # Handle image file
            image_blob = request.files['image'].read()
            ocr_result = detect_text(image_blob)
            return jsonify({'ocr_text': ocr_result[0]})
        
        elif 'pdf' in request.files:
            # Handle PDF file
            pdf_blob = request.files['pdf'].read()
            images = []
            with fitz.open(stream=pdf_blob, filetype="pdf") as pdf:
                for page_num in range(len(pdf)):
                    page = pdf.load_page(page_num)
                    # Convert PDF page to image blob
                    image_bytes = page.get_pixmap().tobytes()
                    images.append(image_bytes)
            
            # Process OCR for each image blob
            ocr_texts = [detect_text(image) for image in images]
            return jsonify({'ocr_texts': ocr_texts})
        
        else:
            return jsonify({'error': 'No supported file part in the request.'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        

# if _name_ == '_main_':
#     app.run(debug=True, port=6000)

if __name__ == '__main__':
    app.run(debug=True)







