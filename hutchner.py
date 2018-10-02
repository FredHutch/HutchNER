#!flask/bin/python
''' author@wlane '''
# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import requests
import sys
from flask import Flask, make_response, jsonify, request, render_template, g, json
from os.path import isfile, join

from sklearn.externals import joblib

from Dates import date_finder
from LSTMExec.model import Model
from Pipelines import ner_negation, ner, general_ner
from flask_oauthlib.provider import OAuth2Provider
import en_core_web_sm

####################
## Preload Models ##
####################
def load_lstm_model(model_dir):
    model = Model(model_path=model_dir)
    # Load existing model
    print ("Loading model...")
    parameters = model.parameters

    # Load reverse mappings
    word_to_id, char_to_id, tag_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
    ]

    # Load the model
    _, f_eval = model.build(training=False, **parameters)
    model.reload()
    return {"model":model,
            "f_eval":f_eval,
            "word_to_id":word_to_id,
            "char_to_id":char_to_id,
            "tag_to_id":tag_to_id,
            "parameters":parameters}

# initialize large models on server startup
spacy_model = en_core_web_sm.load()
lstm_ner_model= load_lstm_model(model_dir=os.path.join(os.path.dirname(__file__), os.path.join("LSTMExec","models","i2b2_fh_50_newlines")))
crf_ner_model= joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("NERResources","Models", "model-test_problem_treatment.pk1")))
crf_deid_model = joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("NERResources","Models",
                                                                                  "model-phone_number_url_or_ip_age_profession_ward_name_employer_email_medical_record_number_account_number_date_provider_name_address_and_components_patient_or_family_name_hospital_name.pk1")))
breast_path_model = joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("NERResources","Models","model-positivesentinelnodes_totalnonsentinelnodes_laterality_tubuleformation_sectiondesc_pathstaget_her2ihc_highriskfinding_pathspecimentype_pathdate_malignantfinding_er_pathsite_benignfinding.pk1")))
breast_laterality_model = joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("NERResources","Models","model-na_right_bilateral_unknown_left.pk1")))

## IMPORTANT: if the model is an LSTM model, "lstm" MUST be found in the key name somewhere, otherwise crf is assumed
models={"crf_ner":crf_ner_model, "lstm_ner":lstm_ner_model, "spacy":spacy_model, "deid_crf":crf_deid_model, "breast_laterality_model": breast_laterality_model, "breast_path_model":breast_path_model}

# configs for CSS colors and headers etc
configs = json.load(open(os.path.join(os.path.dirname(__file__),'css_configs.json'),'r'))

app = Flask(__name__)
oauth=OAuth2Provider(app)

#################
### Endpoints ###
#################
@app.route('/gen_ner/', methods=['GET'])
def general_ner_pipeline():
    documents = request.json
    if documents:
        dt = date_finder.DateFinder()
        for doc_id, doc_txt in documents.items():

            for actual_date_string, indexes, captures in dt.extract_date_strings(doc_txt):
                #logger.debug("Str: {}, idx: {}".format(actual_date_string, indexes))
                p=0
        json_response = general_ner.main(documents, models['spacy'])
        return json_response.encode('utf-8')
    else:
        return make_response(jsonify({'error': 'No data provided'}), 400)


@app.route('/ner/<string:alg_type>', methods=['GET'])
def ner_pipeline(alg_type):
    sys.stdout.write('in ner for alg type')
    documents = request.json
    sys.stdout.write(str(type(documents)))
    if documents:
        json_response = ner.main(documents, alg_type, models)
        return json_response.encode('utf-8')
    else:
        return make_response(jsonify({'error': 'No data provided'}), 400)


@app.route('/ner_neg/<string:alg_type>', methods=['GET'])
def ner_negation_pipeline(alg_type, data=None):
    documents = data or request.json
    if documents:
        json_response = ner_negation.main(documents, alg_type, models)
        return json_response.encode('utf-8')
    return make_response(jsonify({'error': 'No data provided'}), 400)


@app.route('/section_detection', methods = ['GET'])
def section_detection_pipeline():
    return jsonify({"NotImplementedError": "Section detection endpoint is not yet hooked up. Sorry!"})


@app.errorhandler
def not_found():
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.errorhandler(401)
def not_found(error):
    return make_response(jsonify({'401 error':error}),401)

#########################
### HutchNER Demo App ###
#########################


@app.route('/demo/')
def index():
    return render_template('index.html', **locals())


# @app.route('/demo/', methods=['POST', 'GET'])
# def submit_textarea():
#     url = 'https://nlp-brat-prod01.fhcrc.org/hutchner/ner_neg/'
#     json_request_data = json.loads(request.data)
#     algo_type = json_request_data['algo']
#     url += algo_type
#     data={"1":json_request_data['text']}
#     headers = {"content-type": "application/json"}
#     response = requests.get(url, json=data, headers=headers)
#     print response
#     if response.ok:
#         p_response = json.loads(response.text)
#         return json2html(p_response, algo_type)
#     return response.text


@app.route('/demo/', methods=['POST', 'GET'])
def submit_textarea():
    json_request_data = json.loads(request.data)
    algo_type = json_request_data['algo']
    data={"1":json_request_data['text']}
    response = ner_negation_pipeline(algo_type, data)
    p_response = json.loads(response)
    return json2html(p_response, algo_type)


def load_data(data_dir):
    data=dict()
    onlyfiles = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    for file in onlyfiles:
        with open(os.path.join(data_dir, file), "rb") as f:
            text = f.read()
            data[file]=text
    return data


def json2html(json, algo):
    '''
    Converts HutchNER's JSON response into a renderable HTML page

    Make sure if you add new models that you add tag color entries to the css_configs.json file
    otherwise HitchNER Demo will not be able to render the results
    :param json: a JSON object retrieved from HutchNER API
    :param algo: the particular algorithm model called to produce this output
    :return: a string containing the HTML rendering of the data
    '''
    # the key or header for the algorithm/entities
    header = configs.get(algo).get("header")
    # highlighting schema for the entities
    entities = configs.get(algo).get("entities")
    return _render(header, json, entities)


def _render(header, json, colors):
    '''
    This method renders the highlighted concept spans on the HutchNER front end 
    '''
    tokens = json['1']['NER_labels']
    in_span = False
    for token in tokens:
        # If we have not previously opened up a highlight tag
        if not in_span:
            # ... and if the label of the current toke is not "O" (ie should be highlighted)
            if token['label'] != "O":
                # ... then indicate that we have opened up a highlight tag, and actually open the highlight tag in html
                in_span = True
                header += "<span type=\"" + token['label'] + "\" style=\"background-color:" + colors[
                    token['label'].lower()] + insert_negation_color(token) + "\">"
        # If we have already previously opened up a highlight tag
        if in_span:
            # ... and tyhe label of the current token indicates that the token should not be highlighted
            if token['label'] == "O":
                # ... then close the span in html before you add the token text to the string
                in_span = False
                # this gets rid of the problem of highlighting the extra space at the end of the entity
                header = header.rstrip() + "</span> "
        header += token['text'] + " "
        # Whenever you see a period, insert an html newline character
        if token['text'] == ".":
            header += "<br>"
    return header


def insert_negation_color(token):
    if "negation" in token:
        if "DEFINITE" in token["negation"]:
            return ";color:#800000"
        elif "AMBIVALENT" in token["negation"]: # we call this 'EQUIVOCAL' on the front end
            return ";color:#474646"
        else:   # this is 'PROBABLE'
            return ";color:#014070"
    else:
        return ";color:#000000"


if __name__ == '__main__':
    app.run()
