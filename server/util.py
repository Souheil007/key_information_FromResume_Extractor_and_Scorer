import joblib
import json
import numpy as np
import base64
import re 
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import re
import pandas as pd
import docx2txt
import keras.backend as kerback
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda
from keras.models import load_model
from tensorflow.keras.layers import Layer
from datetime import datetime
from dateutil import relativedelta
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
from spacy.matcher import PhraseMatcher
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# # load default skills data base
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import keras
from keras_preprocessing.sequence import pad_sequences
import torch
import torch.nn.functional as F
from io import BytesIO
import spacy
nlp = spacy.load("en_core_web_sm")
Clusters_list = ["Programming Languages and Frameworks", "Web Development", "Backend Development", "Frontend Development", "Database Management", "Networking", "Cybersecurity", "Cloud Computing Platforms", "Containerization and Orchestration", "DevOps Tools", "Operating Systems", "Mobile Development", "Data Science", "Data Visualization", "Artificial Intelligence and NLP", "Testing and Quality Assurance", "Version Control Systems", "Agile Methodologies", "UI/UX Design","Project Management"]
#from keras.preprocessing.sequence import pad_sequences
__model=None
__tokenizer = None
tag2idx={'I-NAME': 0,
 'L-CLG': 1,
 'L-SKILLS': 2,
 'B-EMAIL': 3,
 'L-YOE': 4,
 'I-COMPANY': 5,
 'I-EMAIL': 6,
 'I-DEG': 7,
 'U-GRADYEAR': 8,
 'L-EMAIL': 9,
 'B-GRADYEAR': 10,
 'X': 11,
 'L-NAME': 12,
 'L-GRADYEAR': 13,
 'I-LOC': 14,
 'U-EMAIL': 15,
 '[SEP]': 16,
 'B-CLG': 17,
 'B-DESIG': 18,
 'L-DEG': 19,
 'B-YOE': 20,
 'U-YOE': 21,
 'I-GRADYEAR': 22,
 'U-CLG': 23,
 'B-SKILLS': 24,
 'B-LOC': 25,
 'B-COMPANY': 26,
 'U-SKILLS': 27,
 'L-DESIG': 28,
 'U-COMPANY': 29,
 'U-LOC': 30,
 'B-DEG': 31,
 'U-NAME': 32,
 '[CLS]': 33,
 'U-DEG': 34,
 'U-DESIG': 35,
 'I-CLG': 36,
 'O': 37,
 'L-COMPANY': 38,
 'B-NAME': 39,
 'I-YOE': 40,
 'L-LOC': 41,
 'I-SKILLS': 42,
 'I-DESIG': 43
 }
idx2tag={0: 'I-NAME',
 1: 'L-CLG',
 2: 'L-SKILLS',
 3: 'B-EMAIL',
 4: 'L-YOE',
 5: 'I-COMPANY',
 6: 'I-EMAIL',
 7: 'I-DEG',
 8: 'U-GRADYEAR',
 9: 'L-EMAIL',
 10: 'B-GRADYEAR',
 11: 'X',
 12: 'L-NAME',
 13: 'L-GRADYEAR',
 14: 'I-LOC',
 15: 'U-EMAIL',
 16: '[SEP]',
 17: 'B-CLG',
 18: 'B-DESIG',
 19: 'L-DEG',
 20: 'B-YOE',
 21: 'U-YOE',
 22: 'I-GRADYEAR',
 23: 'U-CLG',
 24: 'B-SKILLS',
 25: 'B-LOC',
 26: 'B-COMPANY',
 27: 'U-SKILLS',
 28: 'L-DESIG',
 29: 'U-COMPANY',
 30: 'U-LOC',
 31: 'B-DEG',
 32: 'U-NAME',
 33: '[CLS]',
 34: 'U-DEG',
 35: 'U-DESIG',
 36: 'I-CLG',
 37: 'O',
 38: 'L-COMPANY',
 39: 'B-NAME',
 40: 'I-YOE',
 41: 'L-LOC',
 42: 'I-SKILLS',
 43: 'I-DESIG'
 }
skills = ["python","java","c++","Networking", "Cybersecurity", "Cloud Computing", "System Administration", "Database Management", "IT Support", "Virtualization", "IT Project Management", "Scripting and Automation", "DevOps", "Backup and Recovery", "Software Development", "IT Compliance and Governance", "Technical Writing", "Troubleshooting", "IT Service Management", "Hardware Installation and Maintenance", "Mobile Device Management", "Web Development", "API Integration", "Data Analysis", "IT Infrastructure Design", "End-User Training", "ITIL", "Agile Methodologies", "Help Desk Support", "Incident Management", "Server Management", "Network Security", "Virtual Private Networks (VPNs)", "Load Balancing", "Firewall Configuration", "Active Directory Management", "Storage Solutions", "Network Protocols (TCP/IP, DNS, DHCP)", "Operating Systems (Windows, Linux, macOS)", "Remote Access Solutions", "Email Systems Management (Exchange, Office 365)", "Monitoring and Logging Tools", "Business Continuity Planning", "Disaster Recovery Planning", "Software Licensing Management", "Configuration Management", "IT Budgeting and Cost Control"]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __model
    global __tokenizer
    if '__model' not in globals() or __model is None:
        # Initialize the model with the same architecture
        __model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tag2idx)) 
     
        __model.load_state_dict(torch.load('./server/artifacts/model_weights2.pth' , map_location='cpu'))
        print("Model loaded successfully.")

        __tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    print("loading saved artifacts...done")

def save_text_as_pdf(text, output_file):
    """
    Save text content as a PDF file.

    :param text: The text content to be saved as PDF.
    :param output_file: The path to save the PDF file.
    """
    with open(output_file, 'wb') as f:
        f.write(text.encode())


def extract_text_from_pdf1(pdf_content):
    '''
    Helper function to extract the plain text from PDF content

    :param pdf_content: Content of the PDF file as a string
    :return: iterator of strings of extracted text
    '''
    for page in PDFPage.get_pages(
                        pdf_content,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        codec='utf-8',
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()
                    yield text

                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            
    return
        
    
def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files

    :param pdf_path: path to PDF file to be extracted (remote or local)
    :return: iterator of string of extracted text
    '''
    # https://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/
    if not isinstance(pdf_path, io.BytesIO):
        # extract text from local pdf file
        with open(pdf_path, 'rb') as fh:
            try:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        codec='utf-8',
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()
                    yield text

                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            except PDFSyntaxError:
                return
    else:
        # extract text from remote pdf file
        try:
            for page in PDFPage.get_pages(
                    pdf_path,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    codec='utf-8',
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield {"content": text} 

                # close open handles
                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            return
def extract_text_from_docx(doc_path):
    '''
    Helper function to extract plain text from .docx files

    :param doc_path: path to .docx file to be extracted
    :return: string of extracted text
    '''
    try:
        temp = docx2txt.process(doc_path)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        return ' '.join(text)
    except KeyError:
        return ' '
def extract_skills1(text):
    nlp = spacy.load("en_core_web_sm")
# # init skill extractor
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    import re
    skills = set()

    content = text
    content = content.replace('\n', ' ')
    content = content.replace('/', ' ')
    content = re.sub(r'\s+', ' ', content)
    content = content.lower()
    annotations = skill_extractor.annotate(text)
    for anno in annotations['results']['full_matches']:
        skills.add(anno['doc_node_value'])
        #print(anno['doc_node_value'])
            
    for anno in annotations['results']['ngram_scored']:
        skills.add(anno['doc_node_value'])
        #print(anno['doc_node_value'])
    return(skills)



    
def extract_skills(text, skills_file=None):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    if not skills_file:
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'skills.csv')
        )
    else:
        data = pd.read_csv(skills_file)
    skills = list(data.columns.values)
    skillset = []
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams
    noun_chunks=[chunk.text.lower().strip() for chunk in doc.noun_chunks]
    #he4a ilemmlek koll kelmtin mab3a4hom wi9arenhom bel les skills
    for i in range(len(noun_chunks) - 1):
        bi_gram = ' '.join(noun_chunks[i:i+2])
        if bi_gram in skills:
            skillset.append(bi_gram)
    
    #he4a ilemmlek koll 3 kelmet mab3a4hom wi9arenhom bel les skills
    for i in range(len(noun_chunks) - 2):
        tri_gram = ' '.join(noun_chunks[i:i+3])
        if tri_gram in skills:
            skillset.append(tri_gram)
    

def extract_text_from_doc(doc_path):
    '''
    Helper function to extract plain text from .doc files

    :param doc_path: path to .doc file to be extracted
    :return: string of extracted text
    '''
    try:
        try:
            import textract
        except ImportError:
            return ' '
        text = textract.process(doc_path).decode('utf-8')
        return text
    except KeyError:
        return ' '

def extract_text_pdf(data):
    text = ''
    for extracted_text in extract_text_from_pdf1(data):
        text += extracted_text + "\n"
    return text


def extract_text(file_path, extension):
    '''
    Wrapper function to detect the file extension and call text
    extraction function accordingly

    :param file_path: path of file of which text is to be extracted
    :param extension: extension of file `file_name`
    '''
    text = ''
    if extension == '.pdf':
        for extracted_text in extract_text_from_pdf(file_path):
            text += extracted_text + "\n"
    elif extension == '.docx':
        text = extract_text_from_docx(file_path)
    elif extension == '.doc':
        text = extract_text_from_doc(file_path)
    return text

def extract_sentences(text):
    # Define a regular expression pattern to split the text into sentences based only on periods
    #sentence_pattern = r"\."  # Only split on periods
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        sentences.append(sent)
    # Split the text into sentences using the regular expression pattern
    #sentences = re.split(sentence_pattern, text)
    
    # Return the extracted sentences
    return sentences

#bech na3mlou tokenization lkoll word
def get_tokenized_train_data(sentences):
    tokenized_texts = []
    for sentence in sentences:
        temp_token = []
        for word in sentence:
            token_list = __tokenizer.tokenize(word.text)  # Assuming __tokenizer is defined elsewhere
            temp_token.extend(token_list)
        tokenized_texts.append(temp_token)
    return tokenized_texts

def get_input_ids(tokenized_texts):
    MAX_LEN = 512
    input_ids = pad_sequences([__tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    return input_ids

def get_attention_masks(input_ids):
    #fel matrice mta3 token_ids i4a ken el valeur >0 te5ou 1
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    return attention_masks

def Embeddingg(Experience_txt,skill_liste):
    word_embd = [0] * 768  # Initialize word embedding
    skill_emb = [0] * 768  # Initialize skill embedding
    text = nlp(Experience_txt)
    #print(i)
    for word in text:
        word_embd = [a + b for a, b in zip(word_embd, embedder.encode(word.text))]
    for skill in skill_liste:
        skill_emb = [a + b for a, b in zip(skill_emb, embedder.encode(skill))]
        v = [a + b for a, b in zip(word_embd, skill_emb)] 
    return v

class ApplyModelLayer(Layer):
    def __init__(self, base_model, num_outputs, **kwargs):
        super(ApplyModelLayer, self).__init__(**kwargs)
        self.base_model = base_model
        self.num_outputs = num_outputs

    def call(self, inputs):
        outputs = [self.base_model(inputs) for _ in range(self.num_outputs)]
        return tf.keras.backend.stack(outputs, axis=1)

    def get_config(self):
        config = super(ApplyModelLayer, self).get_config()
        config.update({
            'base_model': self.base_model,
            'num_outputs': self.num_outputs,
        })
        return config
import random
from keras.models import load_model
# Function to recreate the ApplyModelLayer
def apply_model_layer_from_config(config):
    base_model = config.pop('base_model')
    num_outputs = config.pop('num_outputs')
    return ApplyModelLayer(base_model, num_outputs, **config)
loaded_model = load_model('./server/artifacts/my_model.h5', custom_objects={'ApplyModelLayer': ApplyModelLayer})
dataset=pd.read_json('./server/artifacts/data.json', orient='records')
X = dataset.iloc[:, :768]
X = np.reshape(X, (X.shape[0],X.shape[1]))
sc = MinMaxScaler(feature_range=(-1,1))
X = sc.fit_transform(X)

def prediction(input1):
    L= []
    input1  = sc.transform(input1)
    predicted_Matching_vector = loaded_model.predict(input1)
    for i in range(20):
        liste = []
        for j in range(11):
            if predicted_Matching_vector[0][i][j] != 0:
                liste.append(j)
        #L.append(Clusters_list[i],": Score = ",random.choice(liste),"/10") 
        L.append(f"{Clusters_list[i]} \n {random.choice(liste)}/10")
        #L.append({
        #                'class': f"{Clusters_list[i]}",
        #                'class_probability':random.choice(liste),
#})
    return(L)
def GetScores(txt,liste):
    return prediction(Embeddingg(txt,liste))

def runn(tokenized_textss,input_idss,attention_maskss):
    
        # Set the model to evaluation mode
    __model.eval()

    with torch.no_grad():
        # Convert input tensors to torch tensors
        input_ids_tensor = torch.tensor(input_idss)
        attention_mask_tensor = torch.tensor(attention_maskss)

        # Model prediction
        outputs = __model(input_ids_tensor.to('cpu'), attention_mask=attention_mask_tensor.to('cpu'))
        
    detokenized_text = []
    detokenized_tags = []
    for i in range(len(outputs)):
        predictions = outputs[i]
        predicted_labels = torch.argmax(predictions, dim=-1)
        predicted_tags = [idx2tag[label_id.item()] for label_id in predicted_labels]
        for token, tag in zip(tokenized_textss[i], predicted_tags):
            # If the token is a subword token
            if token.startswith("##"):
                # Append it to the previous token
                detokenized_text[-1] += token[2:]
            else:
                # If it's a new word, append it as it is
                detokenized_text.append(token)
                detokenized_tags.append(tag)
    # Print the detokenized text with corresponding tags
    for token, tag in zip(detokenized_text, detokenized_tags):
        print(f"{token}: {tag}")
    print("detokenization end")
        
        
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    skills = []
    current_skill = ""
    inside_parentheses = False
    for token, tag in zip(detokenized_text, detokenized_tags):
        if inside_parentheses:
            if token == ')':
                inside_parentheses = False
            continue

        if tag.startswith("B-SKILLS"):
            # If the tag indicates the beginning of a skill
            current_skill = token
        elif token in ['(', ',','.']:
            # If the token is '(', ',' or '.'
            # Append the current skill to the list and reset current_skill
            if current_skill:
                skills.append(current_skill)
                current_skill = ""
            if token == '(':
                inside_parentheses = True
        elif tag.startswith("I-SKILLS"):
            # If the tag indicates the continuation of a skill
            current_skill += " " + token
        elif tag.startswith("L-SKILLS"):
            # If the tag indicates the end of a skill
            current_skill += " " + token
            # Append the skill to the list
            skills.append(current_skill)
            # Reset current_skill
            current_skill = ""


    # Print the extracted skills
    for idx, skill in enumerate(skills):
        print(idx, skill)

if __name__ == "__main__":
    load_saved_artifacts()