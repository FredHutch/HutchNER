# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os, io
import time
from functools import reduce

from DataLoading.DataClasses import PredictedAnnotation


class NEREvaluator:
    def __init__(self, tagged_documents, labels):
        print ("initializing evaluation parameters ...")
        self.tagged_documents = tagged_documents #self._dictify_document_list(tagged_documents)
        self.labels = labels
        self.tp_fp_fn_counts_by_tag_exact = self._count_tp_fp_fn_exact(self.tagged_documents)
        self.tp_fp_fn_counts_by_tag_overlap = self._count_tp_fp_fn_overlap(self.tagged_documents)
        self.precision_recall_f1_by_tag_exact = self._calculate_precision_recall_f1(self.tp_fp_fn_counts_by_tag_exact)
        self.precision_recall_f1_by_tag_overlap = self._calculate_precision_recall_f1(self.tp_fp_fn_counts_by_tag_overlap)

    def get_precision_by_tag(self, tag, strict=True):
        if strict:
            return self.precision_recall_f1_by_tag_exact[tag][0]
        else:
            return self.precision_recall_f1_by_tag_overlap[tag][0]

    def get_recall_by_tag(self, tag, strict= True):
        if strict:
            return self.precision_recall_f1_by_tag_exact[tag][1]
        else:
            return self.precision_recall_f1_by_tag_overlap[tag][1]

    def get_f1_by_tag(self, tag, strict = True):
        if strict:
            return self.precision_recall_f1_by_tag_exact[tag][2]
        else:
            return self.precision_recall_f1_by_tag_overlap[tag][2]

    def _calculate_precision_recall_f1(self, tp_fp_fn_counts_by_tag):
        print (str(len(tp_fp_fn_counts_by_tag)) + ' tags in set')
        for k, v in tp_fp_fn_counts_by_tag.items():
            print (str(k) + ' :: ' + str(v))
        print ('')    
        p_r_f1_by_tag = dict()
        for tag in tp_fp_fn_counts_by_tag:
            tp_fp_fn = tp_fp_fn_counts_by_tag[tag]
            try:                
                precision = float(tp_fp_fn["tp"]) / (tp_fp_fn["fp"] + tp_fp_fn["tp"])
            except:
                precision = 0
            try:
                recall = float(tp_fp_fn["tp"]) / (tp_fp_fn["tp"] + tp_fp_fn["fn"])
            except:
                recall = 0
            try:
                f1 = 2 * (precision*recall) / (precision + recall)
            except:
                f1 = 0
            p_r_f1_by_tag[tag] = (precision, recall, f1)
        return p_r_f1_by_tag

    def _count_tp_fp_fn_exact(self, annotated_data):
        tp_fp_fn_counts_by_tag = dict()
        for label in self.labels:
            tp_fp_fn_counts_by_tag[label] = dict()
            tp_fp_fn_counts_by_tag[label]["tp"] = self._count_tp(annotated_data, label, "exact")
            tp_fp_fn_counts_by_tag[label]["fp"] = self._count_fp(annotated_data, label, "exact")
            tp_fp_fn_counts_by_tag[label]["fn"] = self._count_fn(annotated_data, label, "exact")
        return tp_fp_fn_counts_by_tag

    def _count_tp_fp_fn_overlap(self, annotated_data):
        tp_fp_fn_counts_by_tag = dict()
        for label in self.labels:
            tp_fp_fn_counts_by_tag[label] = dict()
            tp_fp_fn_counts_by_tag[label]["tp"] = self._count_tp(annotated_data, label, "overlap")
            tp_fp_fn_counts_by_tag[label]["fp"] = self._count_fp(annotated_data, label, "overlap")
            tp_fp_fn_counts_by_tag[label]["fn"] = self._count_fn(annotated_data, label, "overlap")
        return tp_fp_fn_counts_by_tag

    def _count_tp(self, documents, label, strictness):
        tp = 0
        if strictness == "exact":
            for doc_id, doc in documents.items():
                gold_concepts = []
                predicted_concepts = []
                if label in doc.concepts_gold:
                    gold_concepts = doc.concepts_gold[label]
                predicted_concepts = self.chunk_by_label(doc_id, doc.NER_token_labels, self.labels)[label]
                for gold in gold_concepts:
                    for predicted in predicted_concepts:
                        if predicted.label == label and \
                            gold.stop == predicted.stop and \
                            gold.start == predicted.start and \
                            gold.text in predicted.text:
                            tp += 1
        else:
            for doc_id, doc in documents.items():
                gold_concepts = []
                predicted_concepts = []
                if label in doc.concepts_gold:
                    gold_concepts = doc.concepts_gold[label]
                predicted_concepts = self.chunk_by_label(doc_id, doc.NER_token_labels, self.labels)[label]
                for gold in gold_concepts:
                    for predicted in predicted_concepts:
                        if predicted.label == label and \
                            gold.start <= predicted.stop and  \
                            predicted.start <= gold.stop:
                            tp += 1
        return tp

    def _count_fp(self, documents, label, strictness):
        fp = 0
        if strictness == "exact":
            for doc_id, doc in documents.items():
                gold_concepts = []
                predicted_concepts = []
                if label in doc.concepts_gold:
                    gold_concepts = doc.concepts_gold[label]
                predicted_concepts = self.chunk_by_label(doc_id, doc.NER_token_labels, self.labels)[label]
                for predicted in predicted_concepts:
                    found = False
                    for gold in gold_concepts:
                        if predicted.label == label and \
                            gold.stop == predicted.stop and \
                            gold.start == predicted.start and \
                            gold.text in predicted.text:
                            found = True
                    if not found:
                        fp += 1
        else:
            for doc_id, doc in documents.items():
                gold_concepts = []
                predicted_concepts = []
                if label in doc.concepts_gold:
                    gold_concepts = doc.concepts_gold[label]
                predicted_concepts = self.chunk_by_label(doc_id, doc.NER_token_labels, self.labels)[label]
                for predicted in predicted_concepts:
                    found = False
                    for gold in gold_concepts:
                        if predicted.label == label and \
                            gold.start <= predicted.stop and \
                            predicted.start <= gold.stop:
                            found = True
                    if not found:
                        fp += 1
        return fp

    def _count_fn(self, documents, label, strictness):
        fn = 0
        if strictness == "exact":
            for doc_id, doc in documents.items():
                gold_concepts = []
                predicted_concepts = []
                if label in doc.concepts_gold:
                    gold_concepts = doc.concepts_gold[label]
                predicted_concepts = self.chunk_by_label(doc_id, doc.NER_token_labels, self.labels)[label]
                for gold in gold_concepts:
                    found = False
                    for predicted in predicted_concepts:
                        if predicted.label == label and \
                            gold.stop == predicted.stop and \
                            gold.start == predicted.start and \
                            gold.text in predicted.text:
                            found = True
                    if not found:
                        fn +=1
        else:
            for doc_id, doc in documents.items():
                gold_concepts = []
                predicted_concepts = []
                if label in doc.concepts_gold:
                    gold_concepts = doc.concepts_gold[label]
                predicted_concepts = self.chunk_by_label(doc_id, doc.NER_token_labels, self.labels)[label]
                for gold in gold_concepts:
                    found = False
                    for predicted in predicted_concepts:
                        if predicted.label == label and \
                            gold.start <= predicted.stop and \
                            predicted.start <= gold.stop:
                            found = True
                    if not found:
                        fn +=1
        return fn

    def write_results(self, out_dir, strictness, model_name, string_timestamp):
        
        if strictness == "exact":
            # write exact results
            with open(os.path.join(out_dir,"exact_scores_" + model_name + '_' + string_timestamp + ".txt"), "w", encoding='utf8') as f:
                for label, scores in self.precision_recall_f1_by_tag_exact.items():
                    f.write(label + ":\n")
                    f.write("\tP:\t" + str(scores[0]) + "\n")
                    f.write("\tR:\t" + str(scores[1])+ "\n")
                    f.write("\tF1:\t" +str(scores[2])+ "\n")

        elif strictness == "overlap":
            # write overlap results
            with open(os.path.join(out_dir, "overlap_scores_"+ model_name + '_' + string_timestamp + ".txt"), "w", encoding='utf8') as f:
                for label, scores in self.precision_recall_f1_by_tag_overlap.items():
                    f.write(label + ":\n")
                    f.write("\tP:\t" + str(scores[0])+ "\n")
                    f.write("\tR:\t" + str(scores[1])+ "\n")
                    f.write("\tF1:\t" + str(scores[2])+ "\n")
        else:
            raise ValueError("There was an issue with your designated strictness level: " + strictness +
                             "\n\t Strictness levels must be derived from the domain {'exact', 'overlap'}")
    
    def output_labels(self, out_dir, tagged_documents, model_name, string_timestamp):
        os.mkdir(out_dir + os.path.sep + string_timestamp)
        # output NER labels in brat annotation format for comparison 
        for doc_id in tagged_documents:
            chunk_dict = self.chunk_by_label(doc_id, tagged_documents.get(doc_id).NER_token_labels, self.labels)
            t_idx = 1
            with open(out_dir + os.path.sep + string_timestamp + os.path.sep + doc_id + '.ann','w', encoding='utf8') as anno_out:
                for label in self.labels:
                    for annotation in chunk_dict.get(label):
                        anno_out.write('T' + str(t_idx) + '\t' + label + ' ' + str(annotation.start) + \
                        ' ' + str(annotation.stop) + '\t' + annotation.text + '\n')
                        t_idx += 1
            # assume unix style line endings (this is what SpaCY is assuming in char offset tracking)
            text_out = io.open(out_dir + os.path.sep + string_timestamp + os.path.sep + doc_id + '.txt', 'w', newline='\n', encoding='utf8')
            text_out.write(tagged_documents[doc_id].text)


    def chunk_by_label(self, doc_id, NER_token_labels, labels):
        label_annot_dict = dict()
        # initialize dict with {key=label:value=list()}
        for label in labels:
            label_annot_dict[label] = list()

        for label in labels:
            i=0
            while i < len(NER_token_labels):
                chunk = list()
                try:
                    while NER_token_labels[i]['label'] == label:
                        chunk.append(NER_token_labels[i])
                        i +=1
                except:
                    print ("WARNING - passing on out of range index " + str(i) + \
                        " in document " + str(doc_id) + ". Last token " + \
                        str(NER_token_labels[i-1]) + " added to chunk " + str(chunk[-1].get('label', 'UNK')))
                annotation = self.create_annot_from_chunk(chunk, label)
                if annotation:
                    label_annot_dict[label].append(annotation)
                i += 1
        return label_annot_dict


    def create_annot_from_chunk(self, chunk, label):
        if len(chunk)==0:
            return None

        global_begin =float('inf')
        global_end = float('-inf')
        full_string = ' '.join([x['text'] for x in chunk])
        confidence = list()

        for token in chunk:
            if token['start'] <= global_begin:
                global_begin=token['start']
            if token['stop'] >= global_end:
                global_end = token['stop']
            confidence.append(token['confidence'])

        confidence_avg = reduce(lambda x, y: x + y, confidence) / len(confidence) # calculate avg

        return PredictedAnnotation(label,
                                   global_begin,
                                   global_end,
                                   full_string,
                                   doc_id=None,
                                   prediction_confidence=confidence_avg)
