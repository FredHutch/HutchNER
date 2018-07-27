# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os
import time

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

# initialize large models on server startup
spacy_model = en_core_web_sm.load()
lstm_ner_model= load_lstm_model(model_dir=os.path.join(os.path.dirname(__file__), os.path.join("..","LSTMExec","models","i2b2_fh_50_newlines")))
crf_ner_model= joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("..","NERResources","Models", "model-test_problem_treatment.pk1")))
crf_deid_model = joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("..","NERResources","Models","model-phone_number_url_or_ip_age_profession_ward_name_employer_email_medical_record_number_account_number_date_provider_name_address_and_components_patient_or_family_name_hospital_name.pk1")))
breast_path_model = joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("..","NERResources","Models","model-positivesentinelnodes_totalnonsentinelnodes_laterality_tubuleformation_sectiondesc_pathstaget_her2ihc_highriskfinding_pathspecimentype_pathdate_malignantfinding_er_pathsite_benignfinding.pk1")))
breast_laterality_model = joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("..","NERResources","Models","model-na_right_bilateral_unknown_left.pk1")))



models={"breast_path_model":breast_path_model}

def main():
    """ Entry point to HutchNER1: Concept NERExtraction Training """
    # start timer
    start = time.clock()

    # Parse incoming cmd line arguments
    args = ArgumentParsingSettings.get_testing_args()
    data_dir = args.datadir
    local_annotations = args.annots
    labkey_ini_section = args.section

    # Load the documents
    text_dl = i2b2DataLoader(txt_dir = data_dir, annotation_dir=local_annotations)
    docs = text_dl.load()

    # Run NER driver with models and data provided in dirs
    extractor = NERExtraction(docs, models.keys()[0])
    tagged_documents = extractor.tag_all(models=models)
    neg_documents = extractor.remove_negated_concepts(tagged_documents)

    # Evaluate the performance on TAGGED DOCUMENTS (not the negated ones)
    labels = extractor.possible_labels
    ev = NEREvaluator(tagged_documents, labels)
    ev.write_results("../NEREvaluation/EvalResults", strictness="exact")
    ev.write_results("../NEREvaluation/EvalResults", strictness="overlap")

    # Print time elapsed to console
    end = time.clock()
    print ("##################################")
    print (" \tTime Elapsed: " + str(int((end-start)/60)) + " minutes and " + str(int((end-start) % 60)) + " seconds.")
    print ("##################################")

if __name__ == '__main__':
    main()
