from flask import Flask, request, jsonify
from flask_cors import CORS

import pdfplumber
#from transformers import AutoTokenizer, AutoModelForTokenClassification
#from transformers import pipeline
import spacy 
import re

app = Flask(__name__)
CORS(app)  # Allow all origins

'''
# Load the NLP model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

#ner_model = pipeline("ner", model=model, tokenizer=tokenizer)
# ner_model= pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')
'''

nlp = spacy.load("en_core_web_lg")

@app.route('/upload-pdf', methods=['POST'])
def extract_data():
    file = request.files['file']
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    phone_pattern = r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    # Apply NER on the extracted text
    entries = text.split('\n')
    people_data = []

    for entry in entries:
            # Apply spaCy's NLP pipeline
            doc = nlp(entry)
            person_data = {'Name': '', 'Phone': '', 'Address': '', 'Other': ''}

            # Extract Name and Address
            for ent in doc.ents:
                if ent.label_ == "PERSON" and not person_data['Name']:
                    person_data['Name'] = ent.text
                elif ent.label_ in ["GPE", "LOC"] and not person_data['Address']:
                    person_data['Address'] = ent.text

            # Extract Phone Number using regex
            phone_match = re.search(phone_pattern, entry)
            if phone_match:
                person_data['Phone'] = phone_match.group()

            # Include remaining text as 'Other' if not categorized
            if not person_data['Name'] or not person_data['Address']:
                person_data['Other'] = entry

            # Add to list if it contains data
            if any(person_data.values()):
                people_data.append(person_data)
    print(people_data)
    return jsonify(people_data)

   
    

if __name__ == '__main__':
    app.run(debug=True)
