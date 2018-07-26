#!/bin/bash

python ./train_local.py -d ~/webroot/hutchner/NERResources/Data/2010_concepts_plusFH/combined/txt -a ~/webroot/hutchner/NERResources/Data/2010_concepts_plusFH/combined/concept

sudo service apache2 reload
