# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os

import en_core_web_sm
from NERPreprocessing.DocumentPreprocessing import bratDocumentPreprocessor
from DataLoading.DataClasses import GoldAnnotation
from DataLoading.TextDataLoading import TextDataLoader


class bratDataLoader(TextDataLoader):
    def __init__(self, txt_dir, annotation_dir = None):
        super(bratDataLoader, self).__init__(txt_dir)
        self.annotation_dir = annotation_dir
        self.detected_labels = set()
        self.spacy_model = en_core_web_sm.load()
        self.sent_dict = {}


    def load(self):
        '''
        Using attributes from self, loads documents and annotations in i2b2 format from local dirs into memory
        :return: Document Objs 
        '''
        docs = self.load_documents()
        # Sentence segmentation, tokenization, POS, dep parsing, etc
        bratDocumentPreprocessor(docs, self.spacy_model)
        # Add annotations to document objects
        if self.annotation_dir:
            self.join_annotations(docs)
        return docs

    def get_annotations(self):
        '''
        Public-facing annotation getter. This should be used as info only, not for processing, as annotation formats
        differ depending on the source so we cannot garentee any standard format at tis point.
        The document objects retrieved by get_docs() will contain standardized annotations for downstream
        processing.
        :return: A list of dictionaries of {attrib_type<string>:value<string|int>}
        '''
        if self.annotations:
            return self.annotations
        else:
            raise ValueError("There were no annotations retrieved in dataloading, so 'get_annotations()' returns nothing")

    def _get_annotations(self):
        '''
        Loads just the annotations from brat file
        :return: List of GoldAnnotation objects
        '''
        document_sentidx_data = dict()
        for filename in os.listdir(self.annotation_dir):
            with open(os.path.join(self.annotation_dir, filename), "r") as f:
                annot_lines = f.readlines()
                doc_id = filename.split(".")[0]
                if doc_id not in document_sentidx_data:
                    document_sentidx_data[doc_id] = list()
                for line in annot_lines:
                    annotation  = self._parse_brat_annotation(line)
                    if annotation:
                        tag = annotation[2]
                        self.detected_labels.add(tag)
                        document_sentidx_data[doc_id].append((annotation))
                    else:
                        self.logger.warning("Line in brat annotation with doc id: {} did not fit in current parsing scheme".format(doc_id))
                        pass # this line in the brat file did not fit the current parsing schema
                self.logger.info("annotation with doc id: {} had {} total parsed lines out of {}".format(doc_id, len(document_sentidx_data[doc_id]), len(annot_lines)))
        return document_sentidx_data

    def join_annotations(self, docs):
        '''
        Loads just the annotations from brat file
        :return: List of GoldAnnotation objects
        '''
        self.docs = docs
        if self.annotation_dir:
            self.annotations = self._get_annotations()
            for doc_id, anns in self.annotations.items():    
                self.add_annotation(doc_id, anns)
        return self.docs

    def _parse_brat_annotation(self, line):
        concept = line.strip().split("\t")
        # temporary fix for non entity tags (skips over attribute, relation annotations)
        if len(concept) > 2:
            label_location = concept[1].split()
            iob_class = label_location[0].lower()   # is this lowering of class labels really necessary? (ets)
            start_idx = int(label_location[1])
            end_idx = int(label_location[2])
            text = concept[-1]
            return (start_idx, end_idx, iob_class, text)
        self.logger.info("non-entity tag found in line: {}".format(concept))

    def add_annotation(self, doc_id, annotations):
        doc = self.docs[doc_id]
        for anno in annotations:
            start_offset, end_offset, tag, text = anno
            sent_order_idx = self._get_sent_idx(doc, start_offset, end_offset, text)
            e = GoldAnnotation(tag, start_offset, end_offset, text, sent_order_idx)
            doc.concepts_gold[tag].append(e)
        self.logger.info("brat annotation with doc id: {} added {} annotations".format(doc_id, len(annotations)))
        return True
    

    def _get_sent_idx(self, doc, start_offset, end_offset, text):
        for sent in doc.sentences:
            index = sent.sent_order_idx
            offset_start = sent.span_start
            offset_stop = sent.span_end
            if start_offset >= sent.span_start:
                if start_offset <= sent.span_end:
                    return index
        # default to skipping the annotation if it crosses sentence boundaries
        self.logger.error('ERROR: no sentence bounds found for ' + str(start_offset) + ' to ' + str(end_offset) + '  text: ' + doc.text[start_offset:end_offset])
