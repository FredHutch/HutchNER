# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import time

from DataLoading.i2b2DataLoading import i2b2DataLoader
from DataLoading.bratDataLoading import bratDataLoader

from NERExtraction.Training import NERTrainer
from NERUtilities import ArgumentParsingSettings
import os

def main():
    """ Entry point to GP Concept NERExtraction Training System """
    # start timer
    start = time.clock()

    # Parse incoming cmd line arguments
    args = ArgumentParsingSettings.get_training_args()
    text_dir = args.textdir
    local_annotations = args.annots
    model_name = args.model
    algo_type = args.model_type
    anno_type = args.anno_type
    # load and preprocess the data
    if anno_type == 'i2b2':
        text_dl = i2b2DataLoader(txt_dir=text_dir, annotation_dir=local_annotations, encoding="ISO-8859-1")
    else:
        text_dl = bratDataLoader(txt_dir=text_dir, annotation_dir=local_annotations, encoding="ISO-8859-1")


    #brat_dl = i2b2DataLoader(text_dir, local_annotations)
    docs = text_dl.load()
    detected_labels = text_dl.get_detected_labels()
    

    trainer = NERTrainer(docs, detected_labels, model_name, algo_type)
    model_name = trainer.train()

    end = time.clock()
    print ("##################################")
    print (" Training summary:\n\t 1 model trained")
    print (" \tTime Elapsed: " + str(int((end-start)/60))+ " minutes and " + str(int((end-start)%60)) + " seconds.")
    print ("\tModel written to " + model_name)
    print ("##################################")

if __name__ == '__main__':
    main()
