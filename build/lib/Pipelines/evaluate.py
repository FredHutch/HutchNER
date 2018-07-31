# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os
import time
import pickle

import en_core_web_sm
from sklearn.externals import joblib

from DataLoading.i2b2DataLoading import i2b2DataLoader
from LSTMExec.model import Model
from NEREvaluation.Evaluation import NEREvaluator
from NERExtraction.Extraction import NERExtraction
from NERUtilities import ArgumentParsingSettings

from NERUtilities.DocumentPrinter import HTMLPrinter

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


def load_pickle(file_name):
    model = None
    with open(file_name, 'rb') as f:
        model = pickle.load(f, encoding='bytes') 
    return model



# initialize large models on server startup
spacy_model = en_core_web_sm.load()
lstm_ner_model= load_lstm_model(model_dir=os.path.join(os.path.dirname(__file__), os.path.join("..","LSTMExec","models","i2b2_fh_50_newlines")))
crf_ner_file = os.path.join("NERResources","Models", "model-test_problem_treatment.pk1")
crf_deid_file  = os.path.join("NERResources","Models","model-phone_number_url_or_ip_age_profession_ward_name_employer_email_medical_record_number_account_number_date_provider_name_address_and_components_patient_or_family_name_hospital_name.pk1")
breast_path_ner_file  = os.path.join("NERResources","Models","breast_path_ner.pkl")
breast_laterality_file = os.path.join("NERResources","Models","breast_path_laterality.pkl")

crf_ner_model = load_pickle(crf_ner_file)
crf_deid_model = load_pickle(crf_deid_file)
breast_path_ner_model = load_pickle(breast_path_ner_file)
breast_path_laterality_model = load_pickle(breast_laterality_file)

models = {"breast_path_laterality":breast_path_laterality_model}

def main():
    """ Entry point to HutchNER1: Concept NERExtraction Training """
    # start timer
    start = time.clock()

    # Parse incoming cmd line arguments
    args = ArgumentParsingSettings.get_testing_args()
    text_dir = args.textdir
    local_annotations = args.annots
    labkey_ini_section = args.section
    model_name = args.model
    model_type = args.model_type
    # Load the documents
    text_dl = i2b2DataLoader(txt_dir=text_dir, annotation_dir=local_annotations)
    docs = text_dl.load()

    # Run NER driver with models and data provided in dirs
    extractor = NERExtraction(docs, model_name, model_type)
    tagged_documents = extractor.tag_all(models=models)    
    neg_documents = extractor.remove_negated_concepts(tagged_documents)

    
    
    # Evaluate the performance on TAGGED DOCUMENTS (not the negated ones)
    labels = extractor.possible_labels
    ev = NEREvaluator(tagged_documents, labels)
    ev.output_labels("OutputLabels", tagged_documents, model_name)
    ev.write_results("EvalResults", strictness="exact", model_name=model_name)
    ev.write_results("EvalResults", strictness="overlap", model_name=model_name)

    # Print time elapsed to console
    end = time.clock()
    print ("##################################")
    print (" \tTime Elapsed: " + str(int((end-start)/60)) + " minutes and " + str(int((end-start) % 60)) + " seconds.")
    print ("##################################")

if __name__ == '__main__':
    main()
