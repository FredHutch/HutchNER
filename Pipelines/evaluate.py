# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os
import time
import pickle
import datetime

import en_core_web_sm
from sklearn.externals import joblib

from DataLoading.i2b2DataLoading import i2b2DataLoader
from DataLoading.bratDataLoading import bratDataLoader

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
crf_phi_file  = os.path.join("NERResources","Models","phi.pkl")
#breast_path_ner_file2  = os.path.join("NERResources","Models","breast_path_ner2.pkl")
breast_path_ner_file  = os.path.join("NERResources","Models","breast_path_ner.pkl")
breast_laterality_file = os.path.join("NERResources","Models","breast_path_laterality.pkl")
sectioning_oncology_file = os.path.join("NERResources","Models","sectioning_oncology.pkl")
sectioning_procedure_file = os.path.join("NERResources","Models","sectioning_procedure.pkl")
sectioning_ed_file = os.path.join("NERResources","Models","sectioning_ed.pkl")
sectioning_ltfu_file = os.path.join("NERResources","Models","sectioning_ltfu.pkl")
sectioning_discharge_summary_file = os.path.join("NERResources","Models","sectioning_discharge_summary.pkl")
pinz_ner_file = os.path.join("NERResources","Models","pinz_ner.pkl")


crf_ner_model = load_pickle(crf_ner_file)
crf_phi_model = load_pickle(crf_phi_file)
pinz_ner_model = load_pickle(pinz_ner_file)
breast_path_ner_model = load_pickle(breast_path_ner_file)
breast_path_laterality_model = load_pickle(breast_laterality_file)
sectioning_oncology_model = load_pickle(sectioning_oncology_file)
sectioning_procedure_model = load_pickle(sectioning_procedure_file)
sectioning_ed_model = load_pickle(sectioning_ed_file)
sectioning_ltfu_model = load_pickle(sectioning_ltfu_file)
sectioning_discharge_summary_model = load_pickle(sectioning_discharge_summary_file)


models = {"breast_path_ner":breast_path_ner_model, "breast_path_laterality":breast_path_laterality_model, 
         "sectioning_oncology":sectioning_oncology_model, "sectioning_procedure":sectioning_procedure_model, 
          "phi": crf_phi_model,"pinz_ner":pinz_ner_model,"sectioning_discharge_summary":sectioning_discharge_summary_model,
          "sectioning_ltfu":sectioning_ltfu_model, "sectioning_ed":sectioning_ed_model}

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
    anno_type = args.anno_type
    # Load the documents
    if anno_type == 'i2b2':
        text_dl = i2b2DataLoader(txt_dir=text_dir, annotation_dir=local_annotations)
    else:
        text_dl = bratDataLoader(txt_dir=text_dir, annotation_dir=local_annotations)
    docs = text_dl.load()

    # Run NER driver with models and data provided in dirs
    extractor = NERExtraction(docs, model_name, model_type)
    tagged_documents = extractor.tag_all(models=models)    
    neg_documents = extractor.remove_negated_concepts(tagged_documents)

    
    
    # Evaluate the performance on TAGGED DOCUMENTS (not the negated ones)
    labels = extractor.possible_labels
    ev = NEREvaluator(tagged_documents, labels)

    # use timestamp to link output labels and files to output results numbers
    time_stamp = time.time()
    string_timestamp = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d_%H.%M.%S')

    ev.output_labels("OutputLabels", tagged_documents, model_name, string_timestamp)
    ev.write_results("EvalResults", strictness="exact", model_name=model_name, string_timestamp=string_timestamp)
    ev.write_results("EvalResults", strictness="overlap", model_name=model_name, string_timestamp=string_timestamp)

    # Print time elapsed to console
    end = time.clock()
    print ("##################################")
    print (" \tTime Elapsed: " + str(int((end-start)/60)) + " minutes and " + str(int((end-start) % 60)) + " seconds.")
    print ("##################################")

if __name__ == '__main__':
    main()
