from flask import Flask, request, send_file,jsonify,render_template
import re
import base64
import os
import util
from flask_cors import CORS
import os
import shutil
import json
import pandas as pd
import logging
from flask import Flask, jsonify, request, flash, redirect, Response
from pathlib import Path
app = Flask(__name__)
try:
    path = os.path.dirname(os.path.abspath(__file__))
    upload_folder=os.path.join(
    path.replace("/file_folder",""),"tmp")
    os.makedirs(upload_folder, exist_ok=True)
    app.config['upload_folder'] = upload_folder
except Exception as e:
    app.logger.info('An error occurred while creating temp folder')
    app.logger.error('Exception occurred : {}'.format(e))
CORS(app)

@app.route('/')
def home():
    return render_template("app.html")

@app.route('/cv_scorer', methods=['GET', 'POST'])
def cv_scorer():
    data = request.form['text_content']
    # Extract text from the PDF content
    text = util.extract_text_pdf(data)
    print(text)


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    return "hello"

@app.route('/pass', methods=['POST'])
def post():
    #try:
        pdf_file = request.files['file']
        pdf_name = pdf_file.filename
        save_path = os.path.join(app.config.get('upload_folder'), pdf_name)
        pdf_file.save(save_path)
        
        # Getting file size
        file_size = Path(save_path).stat().st_size
        # Extracting text from PDF
        extracted_text = util.extract_text(save_path,'.pdf')
        
        #formatted_text = re.sub(r'\s{2,}', '.', extracted_text)
        #formatted_text = re.sub(r'\s+', '.', extracted_text)
        formatted_text = '.'.join(extracted_text.splitlines())
        formatted_text = re.sub(r'\.{2,}', ' ', formatted_text)
        sentences = util.extract_sentences(formatted_text)
        tokenized_text = util.get_tokenized_train_data(sentences)
        input_ids= util.get_input_ids(tokenized_text)
        attention_mask=util.get_attention_masks(input_ids)
        
        
        
        print(extracted_text)
        print(formatted_text)
        print(sentences)
        print(len(sentences))
        print(tokenized_text)
        print(input_ids[0])
        print(attention_mask[0])
        skills= util.extract_skills1(extracted_text)
        scores = util.GetScores(extracted_text,[list(skills)])
        print(scores)
        #util.runn(tokenized_text,input_ids, attention_mask)
        for i in scores:
            print(i)
        # Remove temporary folder
        #shutil.rmtree(upload_folder)
        os.remove(save_path)
        response = jsonify({'scores': scores})
        response.headers.add('Access-Control-Allow-Origin', '*')
    #except Exception as e:
    #    app.logger.error("An error occurred: {}".format(e))
    #    return jsonify({'error': 'An error occurred while processing the request'}), 500
        return response
if __name__ == "__main__":
    print("Starting Python Flask Server For cv_scorer")
    util.load_saved_artifacts()
    app.run(port=5000)