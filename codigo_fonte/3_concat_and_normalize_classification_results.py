#!/usr/bin/env pythonc
# coding: utf-8
import os
import re
import argparse
from os import listdir
from os.path import isfile, join
from os import path


## MAIN PROGRAM ##

## Normaliza o resultado gerado pelos algoritmos de classificacao

parser = argparse.ArgumentParser(description='Normalize ML algorithms results.')
parser.add_argument('-i', '--input_path', help='Input path to concat CSV files and to normalize them.', required=False)
parser.add_argument('-otf', '--tf_output_file', help='Output file where TF CSV files will be concatenated and normalized.', required=True)
parser.add_argument('-otfidf', '--tfidf_output_file', help='Output file where TF-idf CSV files will be concatenated and normalized.', required=True)
parser.add_argument('-final', '--final', help='Indicates wether is partial or final concatenation. (True = final; False = partial', required=False, default=True)
args = parser.parse_args()

input_path = args.input_path
output_file_tf = args.tf_output_file
output_file_tfidf = args.tfidf_output_file


if str(args.final) == "True":
    ## TF
    with open(output_file_tf, 'w') as outfile:
        outfile.write("dataset;algorithm;threshold;accuracy_bow;f1_bow;precision_pos_bow;recall_pos_bow;precision_neg_bow;recall_neg_bow;accuracy_gboed_freq;f1_gboed_freq;precision_pos_gboed_freq;recall_pos_gboed_freq;precision_neg_gboed_freq;recall_neg_gboed_freq;reclass_freq;reclass_changed_freq;accuracy_gboed_dist;f1_gboed_dist;precision_pos_gboed_dist;recall_pos_gboed_dist;precision_neg_gboed_dist;recall_neg_gboed_dist;reclass_dist;reclass_changed_dist;param1;param2\n")
        files = ["results_bs/2_tf_final_results.csv","results_huliu2004/2_tf_final_results.csv","results_semeval2014/2_tf_final_results.csv","results_semeval2014_laptop/2_tf_final_results.csv","results_semeval2014_restaurant/2_tf_final_results.csv","results_semeval2015/2_tf_final_results.csv","results_semeval2015_hotel/2_tf_final_results.csv","results_semeval2015_laptop/2_tf_final_results.csv","results_semeval2015_restaurant/2_tf_final_results.csv","results_imdb/2_tf_final_results.csv","results_cgu_pedidos_inf_2020_balanceado_com_genericos/2_tf_final_results.csv","results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_tf_final_results.csv"]
        for f in files:
            if path.exists(f):
                f_in = open(f,"r")
                lines = f_in.readlines()
                f_in.close()

                for l in lines:
                    if not re.search(r'dataset', l):
                        l = l.replace('.',',',)
                        l = l.replace('C4,5','C4.5',)
                        outfile.write(l)

    ## TF-idf
    with open(output_file_tfidf, 'w') as outfile:
        outfile.write("dataset;algorithm;threshold;accuracy_bow;f1_bow;precision_pos_bow;recall_pos_bow;precision_neg_bow;recall_neg_bow;accuracy_gboed_freq;f1_gboed_freq;precision_pos_gboed_freq;recall_pos_gboed_freq;precision_neg_gboed_freq;recall_neg_gboed_freq;reclass_freq;reclass_changed_freq;accuracy_gboed_dist;f1_gboed_dist;precision_pos_gboed_dist;recall_pos_gboed_dist;precision_neg_gboed_dist;recall_neg_gboed_dist;reclass_dist;reclass_changed_dist;param1;param2\n")
        files = ["results_bs/2_tfidf_final_results.csv","results_huliu2004/2_tfidf_final_results.csv","results_semeval2014/2_tfidf_final_results.csv","results_semeval2014_laptop/2_tfidf_final_results.csv","results_semeval2014_restaurant/2_tfidf_final_results.csv","results_semeval2015/2_tfidf_final_results.csv","results_semeval2015_hotel/2_tfidf_final_results.csv","results_semeval2015_laptop/2_tfidf_final_results.csv","results_semeval2015_restaurant/2_tfidf_final_results.csv","results_imdb/2_tfidf_final_results.csv","results_cgu_pedidos_inf_2020_balanceado_com_genericos/2_tf_final_results.csv","results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_tf_final_results.csv"]
        for f in files:
            if path.exists(f):
                f_in = open(f,"r")
                lines = f_in.readlines()
                f_in.close()

                for l in lines:
                    if not re.search(r'dataset', l):
                        l = l.replace('.',',',)
                        l = l.replace('C4,5','C4.5',)
                        outfile.write(l)

else:
    files = [os.path.join(input_path,f) for f in listdir(input_path) if isfile(join(input_path, f))]

    ## TF
    with open(output_file_tf, 'w') as outfile:
        outfile.write("dataset;algorithm;threshold;accuracy_bow;f1_bow;precision_pos_bow;recall_pos_bow;precision_neg_bow;recall_neg_bow;accuracy_gboed_freq;f1_gboed_freq;precision_pos_gboed_freq;recall_pos_gboed_freq;precision_neg_gboed_freq;recall_neg_gboed_freq;reclass_freq;reclass_changed_freq;accuracy_gboed_dist;f1_gboed_dist;precision_pos_gboed_dist;recall_pos_gboed_dist;precision_neg_gboed_dist;recall_neg_gboed_dist;reclass_dist;reclass_changed_dist;param1;param2\n")
        for f in files:
            if re.search(r'2\_tf\_.*\.csv', f):
                f_in = open(f,"r")
                lines = f_in.readlines()
                f_in.close()

                for l in lines:
                    if not re.search(r'dataset', l):
                        l = l.replace('.',',',)
                        l = l.replace('C4,5','C4.5',)
                        outfile.write(l)

    ## TF-idf
    with open(output_file_tfidf, 'w') as outfile:
        outfile.write("dataset;algorithm;threshold;accuracy_bow;f1_bow;precision_pos_bow;recall_pos_bow;precision_neg_bow;recall_neg_bow;accuracy_gboed_freq;f1_gboed_freq;precision_pos_gboed_freq;recall_pos_gboed_freq;precision_neg_gboed_freq;recall_neg_gboed_freq;reclass_freq;reclass_changed_freq;accuracy_gboed_dist;f1_gboed_dist;precision_pos_gboed_dist;recall_pos_gboed_dist;precision_neg_gboed_dist;recall_neg_gboed_dist;reclass_dist;reclass_changed_dist;param1;param2\n")
        for f in files:
            if re.search(r'2\_tfidf\_.*\.csv', f):
                f_in = open(f,"r")
                lines = f_in.readlines()
                f_in.close()

                for l in lines:
                    if not re.search(r'dataset', l):
                        l = l.replace('.',',',)
                        l = l.replace('C4,5','C4.5',)
                        outfile.write(l)

exit()