from flask import Flask, jsonify, request,send_file
from flask_cors import CORS
import assemblyai as aai
import pdfplumber
from gtts import gTTS
# from summarydepend import *;
from translate import Translator
import fitz
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import google.generativeai as genai
import os
# from dotenv import load_dotenv
from IPython.display import display
from IPython.display import Markdown
import textwrap3
from PIL import Image

# load_dotenv()



app = Flask(__name__)
CORS(app)
aai.settings.api_key = "44527a6a3d1e42c38a3d7dc6d7dc8cf3"
API = "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"

GOOGLE='AIzaSyDV3rAW4og5T2CU0IS63ohCAY3nK2MhQbA'
genai.configure(api_key=GOOGLE)
gemmodel = genai.GenerativeModel('gemini-pro')

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

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
        text=''
        print("entered to lang-translation")
        input_type=request.form['input_type']
        target_lang=request.form['input_language']
      
        # print(request.form['text'])
        print(target_lang,"printing target lang")
       
        if (input_type == 'text'):
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




def get_paraphrase_response(input_text, num_return_sequences, num_beams):
    max_chunk = 60
    input_text = input_text.replace('\n', " ")
    input_text = input_text.replace('.', '.<eos>')
    input_text = input_text.replace('?', '?<eos>')
    input_text = input_text.replace('!', '!<eos>')
    sentences = input_text.split('<eos>')
    current_chunk = 0
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))
    print(chunks, len(chunks))
    output_chunks = []
    for chunk_id in range(len(chunks)):
        chunk_text = ' '.join(chunks[chunk_id])
        # print(chunk_text)s
        batch = tokenizer([chunk_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(tgt_text)
        output_chunks.append(tgt_text[0])
    print(len(output_chunks))
    final_output = ' '.join(output_chunks)
    return [final_output]


@app.route('/paraphrase',methods=['POST'])

def paraphrase_text():
    try:
        text=''
        num_return_sequences=1
        print("entered to paraphrase")
        input_type=request.form['input_type']
        if(input_type =="text"):
            text=request.form['input_text']
            print(len(text.split()))
            # num_return_sequences = 6

        else:
            input_file=request.files['input_document']
            text=extract_text_from_pdfplumber(input_file)
        print(text)
        num_beams = 40
        
        results=get_paraphrase_response(text,num_return_sequences,num_beams)
        print(results)

        return jsonify({'paraphrased_text':results})

    
    except Exception as e:
        return jsonify({'error': str(e)}), 500











# from flask import Flask, request, jsonify
from google.cloud import vision
import fitz
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
            print('entered')
            # Handle image file
            image_blob = request.files['image'].read()
            ocr_result = detect_text(image_blob)
            return jsonify({'ocr_texts': ocr_result[0]})
        
        elif 'pdf' in request.files:
            ocr_texts=''
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
            # ocr_texts = [detect_text(image) for image in images]
            # print(ocr_texts)
            for image in images:
                ocr_texts+=detect_text(image)[0]+" "

            print(ocr_texts)

            
            return jsonify({'ocr_texts': ocr_texts})
        
        else:
            return jsonify({'error': 'No supported file part in the request.'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/math_exp_solve', methods=['POST'])
def math_exp_solve():
    try:
        prompt = request.form['prompt']
        prompt +=  "You're a mathematical problem solving expert who can solve any mathematical problem, solve the above given mathematical expression in a correct way by thinking step-by-step\n### Make sure to give me output in HTML format but not in markdown format the output should contain only the solving steps.###"
        response = gemmodel.generate_content(prompt)
        resp = response.text.replace("\n", '')
        return jsonify({'gem_response': resp})        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/generate_handwritten_image", methods=["POST"])
def generate_handwritten_image():
    # Get the text from the form data
    text = request.form.get("text")

    # Path of page (background) photo (I have used a blank page)
    BG = Image.open("font/bg.png")
    sheet_width = BG.width
    gap, ht = 0, 0

    for char in text.replace('\n', ' '):
        cases = Image.open('font/{}.png'.format(str(ord(char))))
        BG.paste(cases, (gap, ht))
        size = cases.width
        height = cases.height
        gap += size
        if sheet_width < gap or len(char) * 115 > (sheet_width - gap):
            gap, ht = 0, ht + 140

    # Save the image
    handwritten_image_path = "handwritten_text.png"
    BG.save(handwritten_image_path)

    # Send the image as a response
    return send_file(handwritten_image_path, mimetype='image/png')
        

# if _name_ == '_main_':
#     app.run(debug=True, port=6000)

if __name__ == '__main__':
    app.run(debug=True)







