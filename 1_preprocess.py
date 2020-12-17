#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import csv
import re
import pandas as pd
import numpy as np
import nltk
from unidecode import unidecode
import joblib
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from string import punctuation
import multiprocessing as mp
import time

#nltk.download('punkt')


##--------------------------------------------------------------
### FUNCTIONS ###

def load_csv(input_):
    return pd.read_csv(input_, sep = "|", engine='python')

def database2csv(input_classes,output_path):
    data = pd.DataFrame()
    list_docs = list()
    list_classes= list()
    list_filename= list()
    
    classes = os.listdir(input_classes)
    for class_ in classes:
        doc_label = os.path.join(input_classes, class_)
        docs = os.listdir(doc_label)
        for doc in docs:
            doc_path = os.path.join(doc_label, doc)
            file = open(doc_path, "r", encoding="utf8")
            text = file.read().replace('\n', ' ').replace('\r', ' ')
            list_docs.append(text)
            list_classes.append(class_)
            list_filename.append(doc)
            file.close()
            # print("File: ", doc, "\tclass: ", class_, "\ttext: ", text)
    data['file_name'] = list_filename
    data['classes'] = list_classes
    data['text'] = list_docs
    data.sort_values(by=['file_name'])
    return data


def gboed_terms_2_list(list_file, text):
    list_terms = list()
    f = open(list_file, "r")
    list_lines = f.readlines()
    for t in text['text']:
        for ll in list_lines:
            if(str(ll).__contains__(";")):
                main_term, text = replace_synonyms(ll, text)
                list_terms.append(main_term.rstrip())
            else:
                list_terms.append(ll.rstrip())
    list_terms.sort(key=len, reverse=True)
    return list_terms, text


def text2lower(text):
    return text.lower()


tokenizer = RegexpTokenizer(r'\w+')
def tokenize(text):
    return word_tokenize(text)


def remove_numbers(text):
    return [re.sub("\d+","", ch) for ch in text]


def replace_punct(text):
    text_without_punctuation = " "
    
    sentences = sent_tokenize(text)
    for s in sentences:
        text_tokens = tokenize(s)
        for w in text_tokens:
            for punct in punctuation:
                w = w.replace(punct," ")
            if w != " ":
                text_without_punctuation = text_without_punctuation + " " + w
        text_without_punctuation = text_without_punctuation + " ."
        # print (text_without_punctuation)
    return text_without_punctuation + " "


def clean(tokens):
    result = []
    for token in tokens:
        if (len(token) > 1):
            result.append(unidecode(str(token)))
    return result


def tokenize_clean(text):
    final_tokens = []
    # print(text)
    # Substitui ! e ? por .
    text = re.sub('[\!\?]', '. ', text)
    # Substitui - por espaço
    text = re.sub('\-', ' ', text)
    # Substitui . ou .. ou ... por .
    text = re.sub('(:?\.\s*){2,}', '. ', text)
    # Substitui quebra de linha por .
    text = re.sub('\n\r', '.', text)
    sentences = sent_tokenize(text)
    # print(sentences)
    for s in sentences:
        # print(s)
        tokens = word_tokenize(s)
        # print(tokens)
        for t in tokens:
            if t != ".":
                # remove tokens com menos de 3 caracteres
                t = re.sub('^.{0,2}$', '', t)
                # remove tokens com 10 caracteres ou mais
                t = re.sub('^.{10,}$', '', t)
                # remove pipe
                t = re.sub('|', '', t)
                # remove pontuação de dentro do token
                t = re.sub('[^a-z0-9áàâãéèêíïóôõöúçñ]+', '', t)
                # remove tokens que possuam números
                if not re.search('^[a-záàâãéèêíïóôõöúçñ]+$', t):
                    t = ""
                
            if t != "" or t == ".":
                final_tokens.append(t)
        
        if len(final_tokens) > 0:
            if final_tokens[-1] != ".":
                final_tokens.append(".")
    # print(' '.join(final_tokens))
    return final_tokens


def remove_accents(tokens):
    final_tokens=[]
    for t in tokens:
        final_tokens.append(remove_acento_word(t))
    return final_tokens
def remove_acento_word(word):
    return(unidecode(word))


def remove_stopwords_english(tokens):
    file = open(stopwords_file,"r")
    stopset = file.read().replace("\n"," ")
    return [t for t in tokens if t not in stopset]


def remove_stopwords_portuguese(tokens):
    final = []
    for t in tokens:
        if remove_stopwords_of_a_token(t) != "":
            final.append(t)
    return final
def remove_stopwords_of_a_token(word):
    file = open(stopwords_file,"r")
    stopset = file.read().replace("\n"," ")
    stopset_array = stopset.split(" ")
    found = False
    for sa in stopset_array:
        if re.search(r'\b' + word + r'\b', sa):
            return "" 
    return word


def stem(tokens):
    result = []
    for token in tokens:
        result.append(stem_word(token))
    return result
def stem_word(word):
    stemmer = SnowballStemmer(language)
    if word != []:
        return stemmer.stem(word)
    return ""


def join_sentence(tokens):
    return ' '.join(tokens)


def generate_bow_tf():
    count_vect = CountVectorizer()
    bow_tf_count_vector = count_vect.fit_transform(df['bow_preproc'])
    # dtm_tf = pd.DataFrame(bow_tf_count_vector.toarray(), columns=count_vect.get_feature_names())
    # dtm_tf.to_csv(os.path.join(output_dir,bow_tf_output_file), sep=";", index=None)

    with open(os.path.join(output_dir,"1_representations",bow_tf_output_file), 'w') as out_file:
        list_features = count_vect.get_feature_names()
        for feature in list_features:
            out_file.write(feature)
            if list_features.index(feature) != len(list_features)-1:
                out_file.write(";")
            else:
                out_file.write("\n")

        lines, columns = bow_tf_count_vector.shape
        for l in range(0,lines):
            for c in range(0,columns):
                out_file.write(str(bow_tf_count_vector[l,c]))

                if c != columns-1:
                    out_file.write(";")
                else:
                    out_file.write("\n")


def generate_bow_tfidf():
    count_vect = CountVectorizer()
    bow_tfidf_count_vector = count_vect.fit_transform(df['bow_preproc'])
    tfidf_transformer = TfidfTransformer()
    bow_tfidf = tfidf_transformer.fit_transform(bow_tfidf_count_vector)
    # dtm_tfidf = pd.DataFrame(bow_tfidf.toarray(), columns=count_vect.get_feature_names())
    # dtm_tfidf.to_csv(os.path.join(output_dir,bow_tfidf_output_file), sep=";", index=None)

    with open(os.path.join(output_dir,"1_representations",bow_tfidf_output_file), 'w') as out_file:
        list_features = count_vect.get_feature_names()
        for feature in list_features:
            out_file.write(feature)
            if list_features.index(feature) != len(list_features)-1:
                out_file.write(";")
            else:
                out_file.write("\n")

        lines, columns = bow_tfidf.shape
        for l in range(0,lines):
            for c in range(0,columns):
                out_file.write(str(bow_tfidf[l,c]))

                if c != columns-1:
                    out_file.write(";")
                else:
                    out_file.write("\n")


def tag_gboed_terms(list_file, term_type, text_lst):
    text_result = list()
    f = open(list_file, "r")
    list_lines = f.readlines()
    for t in text_lst:
        sentences = sent_tokenize(t)
        #print(sentences)
        all_text_lst_sentence = ""
        for s in sentences:
            text_mark = s
            # print(text_mark)
            for ll in list_lines:
                ll = ll.replace("\n","")
                ll = ll.lower()
                if(str(ll).__contains__(";")):
                    # print(text_mark)
                    text_mark = check_synonyms(ll, term_type, text_mark)
                else:
                    ll_unidecode = unidecode(ll)
                    text_mark_unidecode = unidecode(text_mark)
                    regex = '(\<' + ll_unidecode + '|' + ll_unidecode + '\>|\<(\w+ )+' + ll_unidecode + '( \w+)+\>)'
                    # print("Regex: ", regex, "\tText:",text_mark)
                    if not re.search(regex, text_mark_unidecode):
                        text_mark = text_mark.replace(' '+ll+' ', " <"+ll+"><"+term_type+"> ")
                        # print ("=> ", regex, " --- ", text_mark)
            all_text_lst_sentence = all_text_lst_sentence + " " + text_mark
            all_text_lst_sentence = all_text_lst_sentence.lstrip()
        # print(all_text_lst_sentence)
        text_result.append(all_text_lst_sentence)
    # print(text_result)
    return text_result


def check_synonyms(line, term_type, text):
    synonyms=str(line).split(";")
    #print(synonyms)
    synonyms_ordered = sorted(synonyms, key=len, reverse=True)
    synonyms_ordered_unidecode = [unidecode(so) for so in synonyms_ordered]
    text = " "+text+" "
    text_unidecode = unidecode(text)
    # print(synonyms_ordered_unidecode)
    for i, sou in enumerate(synonyms_ordered_unidecode[0: len(synonyms_ordered_unidecode)]):
        regex = ' '+sou+' '
        # print ("=> ", regex, " ++++ ", text_unidecode)
        if re.search(regex,text_unidecode):
            regex = '(\<' + sou + '|' + sou + '\>|\<(\w+ )+' + sou + '|' + sou + '( \w+)+\>)'
            # print ("=> ", regex, " ++++ ", text_unidecode)
            if not re.search(regex,text_unidecode):
                text = re.sub('\s(?:'+synonyms_ordered[i]+'|'+synonyms_ordered_unidecode[i]+')\s', " <"+synonyms[0]+"><"+term_type+"> ", text)
                # text = text.replace(' '+synonyms_ordered_unidecode[i]+' ', " <"+synonyms[0]+"><"+term_type+"> ")
                # print ("=====> ", synonyms_ordered_unidecode[i], " --- ", text)
    text = re.sub('^\s+','',text)
    text = re.sub('\s+$','',text)
    return text


def generate_exp_domain_tf_lst():
    exp_pt = list()
    exp_nt = list()
    text_lst = df['gboed_preproc']
    for t in text_lst:
        sentences = sent_tokenize(t)
        for s in sentences:
            dt_list = list()
            pt_list = list()
            nt_list = list()
            terms_findall = re.findall(r'(\<.*?\>\<.*?\>)', s)
            for term in terms_findall:
                dt = re.search(r'\<(.*?)\>\<DT\>', term)
                pt = re.search(r'\<(.*?)\>\<PT\>', term)
                nt = re.search(r'\<(.*?)\>\<NT\>', term)
                if dt:
                    dt = dt.group(1)
                    dt = dt.replace(r' ','_')
                    if not dt in dt_list:
                        dt_list.append(dt)
                elif pt:
                    pt = pt.group(1)
                    pt = pt.replace(r' ','_')
                    if not pt in pt_list:
                        pt_list.append(pt)
                elif nt:
                    nt = nt.group(1)
                    nt = nt.replace(r' ','_')
                    if not nt in nt_list:
                        nt_list.append(nt)

            # Getting Positive Domain Expressions
            for dt in dt_list:
                for pt in pt_list:
                    expr = dt + "_0_" + pt
                    
                    if not expr in exp_pt:
                        exp_pt.append(expr)

            # Getting Negative Domain Expressions
            for dt in dt_list:
                for nt in nt_list:
                    expr = dt+"_1_"+nt
                    
                    if not expr in exp_nt:
                        exp_nt.append(expr)

    exp_pt.sort()
    exp_nt.sort()
    exp_domain = exp_pt + exp_nt
    del(exp_pt)
    del(exp_nt)
    gboed_tf = pd.DataFrame({'exp':exp_domain})
    gboed_tf.T.to_csv(os.path.join(output_dir,"1_representations",gboed_tf_output_file), sep=";", index=None, header=False)
    del(gboed_tf)
    return exp_domain


def generate_gboed_tf():
    exp_domain = generate_exp_domain_tf_lst()
    predict_class_gboed_freq = list()
    text_lst = df['gboed_preproc']
    for t in text_lst:
        tf = list()
        count_pos = 0
        count_neg = 0
        for i in range(0,len(exp_domain)):
            tf.append(0)
        sentences = sent_tokenize(t)
        # print(sentences)
        all_text_lst_sentence = ""
        for s in sentences:
            dt_list = list()
            pt_list = list()
            nt_list = list()
            terms_findall = re.findall(r'(\<.*?\>\<.*?\>)', s)
            # print(">",s)
            for term in terms_findall:
                dt = re.search(r'\<(.*?)\>\<DT\>', term)
                pt = re.search(r'\<(.*?)\>\<PT\>', term)
                nt = re.search(r'\<(.*?)\>\<NT\>', term)
                if dt:
                    dt = dt.group(1)
                    dt = dt.replace(r' ','_')
                    if not dt in dt_list:
                        dt_list.append(dt)
                    # print("DT:",dt)
                elif pt:
                    pt = pt.group(1)
                    pt = pt.replace(r' ','_')
                    if not pt in pt_list:
                        pt_list.append(pt)
                    # print("PT:",pt)
                elif nt:
                    nt = nt.group(1)
                    nt = nt.replace(r' ','_')
                    if not nt in nt_list:
                        nt_list.append(nt)
                    # print("PT:",nt)

            # Getting Positive Domain Expressions
            for dt in dt_list:
                for pt in pt_list:
                    expr = dt+"_0_"+pt
                    
                    if expr in exp_domain:
                        index_exp = exp_domain.index(expr)
                        tf[index_exp] = tf[index_exp] + 1
                        count_pos = count_pos + 1
                        # print("EXP: ", expr)

            # Getting Negative Domain Expressions
            for dt in dt_list:
                for nt in nt_list:
                    expr = dt+"_1_"+nt
                    if expr in exp_domain:
                        index_exp = exp_domain.index(expr)
                        tf[index_exp] = tf[index_exp] + 1
                        count_neg = count_neg + 1
                        # print("EXP: ", expr)

        # print("====>",exp_domain)
        # print("====>",tf)
        if count_pos > count_neg:
            predict_class_gboed_freq.append(positive_class_name)
        elif count_pos < count_neg:
            predict_class_gboed_freq.append(negative_class_name)
        else:
            predict_class_gboed_freq.append(neutral_class_name)

        fd = open(os.path.join(output_dir,"1_representations",gboed_tf_output_file),'a')
        wr = csv.writer(fd, delimiter=";")
        wr.writerow(tf)
        fd.close

    df['gboed_freq_class'] = predict_class_gboed_freq


def generate_exp_domain_dist_lst():
    
    exp_pt = list()
    exp_nt = list()
    text_lst = df['gboed_preproc']
    for t in text_lst:
        # print(">",t)
        sentences = sent_tokenize(t)
        # print(sentences)
        for s in sentences:
            terms_findall = re.findall(r'(\<.*?\>\<.*?\>)', s)
            # print(">",s)

            # Sort terms_findall by <DT>
            r = re.compile(".*<DT>")
            terms_DT = list(dict.fromkeys(list(filter(r.match, terms_findall))))
            r = re.compile("^((?!DT).)*$")
            terms_PTNT = list(dict.fromkeys(list(filter(r.match, terms_findall))))

            
            for term_DT in terms_DT:
                is_pt = False
                for term_PTNT in terms_PTNT:
                    # Building Domain Expressions
                    dt = re.search(r'\<(.*?)\>\<DT\>', term_DT)
                    # print("\tDT:", dt)
                    if dt:
                        #print(term_DT, "\t", dt.group(1))
                        dt = dt.group(1).replace(r' ','_')
                    
                    # Getting Positive Domain Expressions
                    if re.search('.*<PT>', term_PTNT):
                        pt = re.search(r'\<(.*?)\>\<PT\>', term_PTNT)
                        if pt:
                            #print(term_PTNT, "\t", pt.group(1))
                            pt = pt.group(1).replace(r' ','_')
                            #print(dt, "\t", pt)
                            expr = dt+"_0_"+pt
                            is_pt = True
                            
                            if not expr in exp_pt:
                                exp_pt.append(expr)

                    else:
                        # Getting Negative Domain Expressions
                        nt = re.search(r'\<(.*?)\>\<NT\>', term_PTNT)
                        # print("DT: ", dt, "\tNT: ",nt)
                        if nt:
                            #print(term_PTNT, "\t", nt.group(1))
                            nt = nt.group(1).replace(r' ','_')
                            expr = dt+"_1_"+nt
                            is_pt = False

                            if not expr in exp_nt:
                                exp_nt.append(expr)

    exp_pt.sort()
    exp_nt.sort()
    exp_domain = exp_pt + exp_nt
    del(exp_pt)
    del(exp_nt)
    gboed_dist = pd.DataFrame({'exp':exp_domain})
    gboed_dist.T.to_csv(os.path.join(output_dir,"1_representations",gboed_dist_output_file), sep=";", index=None, header=False)
    del(gboed_dist)
    return exp_domain


def generate_gboed_dist():
    exp_domain = generate_exp_domain_dist_lst()
    dist_list = list()
    predict_class_gboed_dist = list()
    text_lst = df['gboed_preproc']
    for t in text_lst:
        # print(">",t)
        dist = list()
        count_pos = 0
        count_neg = 0
        for i in range(0,len(exp_domain)):
            dist.append(0.0)
        
        sentences = sent_tokenize(t)
        # print(sentences)
        for s in sentences:
            terms_findall = re.findall(r'(\<.*?\>\<.*?\>)', s)
            # print(">",s)

            # Sort terms_findall by <DT>
            r = re.compile(".*<DT>")
            terms_DT = list(dict.fromkeys(list(filter(r.match, terms_findall))))
            r = re.compile("^((?!DT).)*$")
            terms_PTNT = list(dict.fromkeys(list(filter(r.match, terms_findall))))

            
            for term_DT in terms_DT:
                is_pt = False
                for term_PTNT in terms_PTNT:
                    regex1 = term_DT + '(.*?)' + term_PTNT
                    r1 = re.search(regex1, s)
                    regex2 = term_PTNT + '(.*?)' + term_DT
                    r2 = re.search(regex2, s)
                    if r1:
                        middle_text = r1.group(1)
                        # print("r1: " + term_DT + " " + term_PTNT, " ---- ",r1.group(1))
                    if r2:
                        middle_text = r2.group(1)
                        # print("r2: " + term_PTNT + " " + term_DT, " ---- ",r2.group(1))

                    # Building Domain Expression
                    # print(middle_text.split())
                    middle_len = len(middle_text.split())
                    if middle_len > 0:
                        middle_dist = 1.0/middle_len
                    else:
                        middle_dist = 1.0

                    dt = re.search(r'\<(.*?)\>\<DT\>', term_DT)
                    # print("\tDT:", dt)
                    if dt:
                        #print(term_DT, "\t", dt.group(1))
                        dt = dt.group(1).replace(r' ','_')
                    
                    # Getting Positive Domain Expressions
                    if re.search('.*<PT>', term_PTNT):
                        pt = re.search(r'\<(.*?)\>\<PT\>', term_PTNT)
                        if pt:
                            #print(term_PTNT, "\t", pt.group(1))
                            pt = pt.group(1).replace(r' ','_')
                            #print(dt, "\t", pt)
                            expr = dt+"_0_"+pt
                            is_pt = True

                    else:
                        # Getting Negative Domain Expressions
                        nt = re.search(r'\<(.*?)\>\<NT\>', term_PTNT)
                        # print("DT: ", dt, "\tNT: ",nt)
                        if nt:
                            #print(term_PTNT, "\t", nt.group(1))
                            nt = nt.group(1).replace(r' ','_')
                            expr = dt+"_1_"+nt
                            is_pt = False

                    

                    if expr in exp_domain:
                        index = exp_domain.index(expr)
                        if(dist[index] < middle_dist):
                            dist[index] = middle_dist
                                                
                        if is_pt:
                            count_pos = count_pos + middle_dist
                        else:
                            count_neg = count_neg + middle_dist

        # print("====>",exp_domain)
        # print("====>",dist)
        dist_list.append(dist)

        if count_pos > count_neg:
            predict_class_gboed_dist.append(positive_class_name)
        elif count_pos < count_neg:
            predict_class_gboed_dist.append(negative_class_name)
        else:
            predict_class_gboed_dist.append(neutral_class_name)

        fd = open(os.path.join(output_dir,"1_representations",gboed_dist_output_file),'a')
        wr = csv.writer(fd, delimiter=";")
        wr.writerow(dist)
        fd.close

    df['gboed_dist_class'] = predict_class_gboed_dist



def job_generate_representations(number):
    print(number[0])
    if(number[0] == 1):
        print('\t> Generating BOW - TF')
        generate_bow_tf()
        print('\t> Done BOW - TF')
    elif(number[0] == 2):
        print('\t> Generating BOW - TFidf')
        generate_bow_tfidf()
        print('\t> Done BOW - TFidf')
    elif(number[0] == 3):
        print('\t> Generating gBoED Freq')
        generate_gboed_tf()
        print('\t> Done')
    elif(number[0] == 4):
        print('\t> Generating gBoED Dist')
        generate_gboed_dist()
        print('\t> Done')


##--------------------------------------------------------------

## MAIN PROGRAM ##

## O diretorio que contem os arquivos de entrada deve possui os arquivos TXT com os documentos
## separados por suas classes. Cada classe é um diretório.
## Ex.: positive / negative

parser = argparse.ArgumentParser(description='Convert TXT documents to a labeled CSV file.')
parser.add_argument('-i', '--input_dir' , required=True, help='Input data directory.')
parser.add_argument('-o', '--output_dir' , required=True, help='Output data directory.')
parser.add_argument('-l', '--language' , required=True, help='Language used in files.')
parser.add_argument('-d', '--database' , required=True, help='Database name.')
parser.add_argument('-sw', '--stopwords' , required=True, help='Stopwords list file.')
parser.add_argument('-dt', '--domain_terms' , required=True, help='Domain terms list file path.')
parser.add_argument('-pl', '--positive_terms' , required=True, help='Positive terms list file path.')
parser.add_argument('-nl', '--negative_terms' , required=True, help='Negative terms list file path.')
parser.add_argument('-pc', '--positive_class' , required=True, help='Positive class name.')
parser.add_argument('-nc', '--negative_class' , required=True, help='Negative class name.')
parser.add_argument('-nnc', '--neutral_class' , required=True, help='Neutral class name.')
args = parser.parse_args()


input_dir = args.input_dir
output_dir = args.output_dir
stopwords_file = args.stopwords
language = args.language
database = args.database.lower()
positive_class_name = args.positive_class
negative_class_name = args.negative_class
neutral_class_name = args.neutral_class
procs = mp.cpu_count()-1


## FILENAMES ##
bow_tf_output_file = "1_bow_tf_" + database + ".csv"
bow_tfidf_output_file = "1_bow_tfidf_" + database + ".csv"
gboed_tf_output_file = "1_gboed_freq_" + database + ".csv"
gboed_dist_output_file = "1_gboed_dist_" + database + ".csv"
preprocess_output_file = "1_preprocess_" + database + ".csv"

if language == "portuguese":
    pos_tagger_ptbr = joblib.load('libs/trained_POS_taggers/POS_tagger_brill.pkl')


#ENTRADA DE DADOS PARA TESTES
# print('\t> Loading data files')
# df = load_csv(os.path.join(output_dir,"1_preprocess_teste.csv"))
# print('\t> Done')


print('> Loading data files')
df = database2csv(input_dir,output_dir)
print('> Done')

# df = df.sample(10)
# df = df[5:6]

print("> Preprocessing TXT files to BOW")
if language == "english":
    df['bow_preproc'] = df['text'].apply(text2lower).apply(replace_punct).apply(tokenize).apply(clean).apply(remove_numbers).apply(remove_stopwords_english).apply(stem).apply(join_sentence)
elif language == "portuguese":
    df['bow_preproc'] = df['text'].apply(text2lower).apply(tokenize_clean).apply(remove_accents).apply(remove_stopwords_portuguese).apply(stem).apply(join_sentence)
print('\t> Done')


print("> Preprocessing TXT files to gBoED")
df['gboed_preproc'] = tag_gboed_terms(args.domain_terms, "DT", df['text'].apply(text2lower).apply(replace_punct))
df['gboed_preproc'] = tag_gboed_terms(args.positive_terms, "PT", df['gboed_preproc'])
df['gboed_preproc'] = tag_gboed_terms(args.negative_terms, "NT", df['gboed_preproc'])
print('> Done')


print("> INICIO DAS REPRESENTACOES")
# 1:BOW TF, 2:BOW TF-idf, 3: gBoED-Freq, 4: gBoED-Dist

print('\t> Generating BOW - TF')
generate_bow_tf()
print('\t> Done BOW - TF')
print('\t> Generating BOW - TFidf')
generate_bow_tfidf()
print('\t> Done BOW - TFidf')
print('\t> Generating gBoED Freq')
generate_gboed_tf()
print('\t> Done')
print('\t> Generating gBoED Dist')
generate_gboed_dist()
print('\t> Done')

df.to_csv(os.path.join(output_dir,preprocess_output_file), sep="|", index=None)