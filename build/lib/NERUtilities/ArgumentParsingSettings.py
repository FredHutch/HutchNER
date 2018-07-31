# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import argparse


def get_training_args():
    """
    Defines the command line arguments necessary to kick off a training session
    :return: the command line arguments provided from stdin
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--textdir", 
                        help="The directory containing your raw training data", required=True)
    parser.add_argument("-m", "--model", 
                        help="The name of the model used for config and pickled model files")
    parser.add_argument("-mt", "--model_type", 
                        help="crf or lstm (defaults to CRF)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("-s", "--section",
                        help="The section of labkey.ini that has the settings to pull data",
                        required=False)
    action.add_argument("-a", "--annots", 
                        help="The path to the directory housing annotations locally")
    
    args = parser.parse_args()
    
    if args.section is None and args.annots is None:
        raise ValueError("You are running a training script, but no but not specified "
                        "the location of the rebel ba-- er, annotation files. Either set -s "
                        "for labkey.ini section, or -a for the local annotation directory.")
    if args.model is None:
        print ("You have not supplied a model name. The name will be generated automatically " + \
                "from the tags available in the annotation file..." + \
                "and it's gonna be all awkward and ugly")
    return args

def get_testing_args():
    """
        Defines the command line arguments necessary to kick off a testing session
        :return: the comman line arguments provided from stdin
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--textdir", 
                        help="The directory of the raw documents you want to tag")
    parser.add_argument("-m", "--model", 
                        help="The name of the model used in training", required=True)
    parser.add_argument("-mt", "--model_type", 
                        help="crf or lstm (defaults to CRF)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("-s", "--section",
                        help="The section of labkey.ini that has the settings to pull data")
    action.add_argument("-a", "--annots", help="The directory housing local annotations")

    args = parser.parse_args()
    return args


def get_local_predict_args():
    """
    Defines the command line arguments necessary to kick off an extraction session
    :return: the command line arguments provided from stdin
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt", "--model_type", help="Choose a model type: either CRF or LSTM. If not specified, default is CRF",
                        default="CRF")
    parser.add_argument("-t", "--textdir", help="The directory location of the raw documents you want to tag with the NER "
                        "system", required=True)
    parser.add_argument("-o", "--output", default="./", help="Output file location")
    parser.add_argument("-m", "--model", help="Name of the model to apply in prediction", required=True)

    args = parser.parse_args()
    return args


def get_labkey_predict_args():
    """
    Defines the command line arguments necessary to kick off an extraction session
    :return: the command line arguments provided from stdin
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt", "--model_type", help="Choose a model type: either CRF or LSTM. If not specified, default is CRF",
                        default="CRF")
    parser.add_argument("-s", "--section",
                        help="The section of the labkey.ini that has the labkey settings you want to use to pull data")
    parser.add_argument("-o", "--output", default="./", help="Output file location")
    parser.add_argument("-m", "--model", help="Name of the model to apply in prediction", required=True)

    args = parser.parse_args()
    return args
