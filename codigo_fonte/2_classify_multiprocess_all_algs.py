#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import csv
import itertools
import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm as SVM
from multiprocessing import Pool
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

##--------------------------------------------------------------
### FUNCTIONS ###


def start_job(alg, bow_metric, dataset, pos_label, neg_label):
    list_param = list()
    if (alg==1):
        print("\n\nExecuting KNN ...")
        if dataset=="SemEval2015_Hotel":
            configs_pd = pd.read_csv("configs_algs/configs_knn_hotel.csv", sep='|')
        else:
            configs_pd = pd.read_csv("configs_algs/configs_knn.csv", sep='|')
        for index, row in configs_pd.iterrows():
            list_param.append((1, bow_metric, row['n'], row['distance'], pos_label, neg_label))

    elif (alg==2):
        print("\n\nExecuting MNB ...")
        configs_pd = pd.read_csv("configs_algs/configs_mnb.csv", sep='|')
        for index, row in configs_pd.iterrows():
            list_param.append((2, bow_metric, row['alpha'], '', pos_label, neg_label))

    elif (alg==3):
        print("\n\nExecuting C4.5 ...")
        configs_pd = pd.read_csv("configs_algs/configs_c45.csv", sep='|')
        for index, row in configs_pd.iterrows():
            list_param.append((3, bow_metric, row['criterion'], '', pos_label, neg_label))

    elif(alg==4):
        print("\n\nExecuting SVM ...")
        configs_pd = pd.read_csv("configs_algs/configs_svm.csv", sep='|')
        for index, row in configs_pd.iterrows():
            list_param.append((4, bow_metric, row['kernel'], row['gamma'], pos_label, neg_label))
    p.map(job, list_param)


def job(parameters):
    alg = parameters[0]
    bow_metric = parameters[1]
    param1 = parameters[2]
    param2 = parameters[3]
    pos_label = parameters[4]
    neg_label = parameters[5]
    
    print("STARTING... ",dict_alg[str(alg)], "-", param1, "-", param2, "-", bow_metric)

    resultados, accuracy_bow, f1_bow, th_folds = [], [], [], []
    precision_pos_bow, precision_neg_bow, recall_pos_bow, recall_neg_bow = [], [], [], []
    count_fold = 0
    
    # OPENNING PRINTING FOLDS
    output_folds_file = os.path.join(output_folds_path, '2_'+bow_metric+'_'+dict_alg[str(alg)]+'_'+str(param2)+'_'+ str(param1)+'.txt')
    f_fold_out = open(output_folds_file, mode='w')
    
    for train_index, test_index in kf.split(data, labels):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        z_test = filenames.iloc[test_index]
        
        # PRINTING FOLDS
        count_fold += 1 
        f_fold_out.write("\n\n\n> INDEXES")
        f_fold_out.write("\n>> FOLD_" + str(count_fold))
        f_fold_out.write("\n\n>> TRAIN INDEX\n")
        for item in train_index:
            f_fold_out.writelines(str(item)+" "+df_input['file_name'][item]+"\n")
        f_fold_out.write("\n\n>> TEST INDEX\n")
        for item in test_index:
            f_fold_out.writelines(str(item)+" "+df_input['file_name'][item]+"\n")
        f_fold_out.close
        
        if (alg == 1):
            if(bow_metric == "tf"):
                text_clf  = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer(use_idf=False)),
                                    ('clf', KNeighborsClassifier( n_neighbors= param1, 
                                                                  metric= param2)), 
                                    ])
            else:
                text_clf  = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer(use_idf=True)),
                                    ('clf', KNeighborsClassifier( n_neighbors= param1, 
                                                                  metric= param2)), 
                                    ])

        elif (alg == 2):
            if(bow_metric == "tf"):
                text_clf  = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer(use_idf=False)),
                                    ('clf', MultinomialNB(alpha = param1)),
                                    ])
            else:
                text_clf  = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer(use_idf=True)),
                                    ('clf', MultinomialNB(alpha = param1)),
                                    ])

        elif (alg == 3):
            if(bow_metric == "tf"):
                text_clf  = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer(use_idf=False)),
                                    ('clf', DecisionTreeClassifier( max_depth=10,
                                                                    random_state=1234,
                                                                    criterion = param1)),
                                    ])
            else:
                text_clf  = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer(use_idf=True)),
                                    ('clf', DecisionTreeClassifier( max_depth=10,
                                                                    random_state=1234,
                                                                    criterion = param1)),
                                    ])

        elif (alg == 4):
            if(bow_metric == "tf"):
                text_clf = Pipeline([('vect', CountVectorizer()),
                                     ('tfidf', TfidfTransformer(use_idf=False)),
                                     ('clf', SVM.SVC(coef0=2.0,
                                                     probability=True,
                                                     kernel=param1,
                                                     gamma=param2,
                                                     class_weight='balanced',
                                                     random_state=1234)),
                                     ])
            else:
                text_clf = Pipeline([('vect', CountVectorizer()),
                                     ('tfidf', TfidfTransformer(use_idf=True)),
                                     ('clf', SVM.SVC(coef0=2.0,
                                                     probability=True,
                                                     kernel=param1,
                                                     gamma=param2,
                                                     class_weight='balanced',
                                                     random_state=1234)),
                                     ])

        # print(y_train.unique())
        text_clf.fit(x_train, y_train)
        y_pred = text_clf.predict(x_test)
        #print(text_clf.predict_proba(X_test))
        resultados_fold_gk = pd.DataFrame()
        resultados_fold_gk['file_name'] = z_test.values
        resultados_fold_gk['real'] = y_test.values
        resultados_fold_gk['predicted'] = y_pred
        print("\n\n\n>ALG:",dict_alg[str(alg)], " - ", param1, " - ", param2, "-", bow_metric)
        print(">FILENAME>\n",resultados_fold_gk['file_name'])
        # print(">RESULTS REAL\n",resultados_fold_gk['real'])
        # print(">RESULTS BOW\n",resultados_fold_gk['predicted'])
        resultados_fold_gk['porcentagem'] = np.max(text_clf.predict_proba(x_test), axis=1)
        # print(">PORCENTAGENS:\n",resultados_fold_gk['porcentagem'])
        resultados_fold_gk['param1'] = param1
        resultados_fold_gk['param2'] = param2
        
        resultados = pd.merge(resultados_fold_gk, df_input, on="file_name")
        # print("RESULTADOS: ", resultados)
        accuracy_bow.append(accuracy_score(y_test, y_pred))
        f1_bow.append(f1_score(y_test, y_pred, pos_label=pos_label, average = 'macro'))
        precision_pos_bow.append(precision_score(y_test, y_pred, pos_label=pos_label, average='macro'))
        recall_pos_bow.append(recall_score(y_test, y_pred, pos_label=pos_label, average='macro'))
        precision_neg_bow.append(precision_score(y_test, y_pred, pos_label=neg_label, average='macro'))
        recall_neg_bow.append(recall_score(y_test, y_pred, pos_label=neg_label, average='macro'))

        th_folds.append(enriquecimento_folds(alg, param1, param2, resultados, pos_label, neg_label))
        # print("LIST LENGHT TH FOLDS: ", len(th_folds))

    gboed_save(pd.concat(th_folds), accuracy_bow, f1_bow, bow_metric, 
            precision_pos_bow, recall_pos_bow, precision_neg_bow, recall_neg_bow)


def enriquecimento_folds(alg_, param1, param2, results_, pos_label, neg_label):
    results_ = results_.drop(['classes'], axis=1)
    lista_porc = list(results_['porcentagem'])
    lista_predic = list(results_['predicted'])
    th = []
    
    for threshold_min in np.arange(0.5, 1.04, 0.05):
    # threshold_min = 0.55
        print("\n\n>THRESHOLD_MIN:",threshold_min)
        results_['enriched_gboed_freq'], reclass_freq, reclass_changed_freq = enriquecimento(lista_porc, lista_predic, list(results_['gboed_freq_class']), threshold_min)
        results_['enriched_gboed_dist'], reclass_dist, reclass_changed_dist = enriquecimento(lista_porc, lista_predic, list(results_['gboed_dist_class']), threshold_min)
        print(">PORCENTAGENS:\n", lista_porc)
        print(results_['file_name'], "\n", results_['real'],"\n" , results_['predicted'], "\n", results_['enriched_gboed_freq'], "\n", results_['enriched_gboed_dist'])
        accuracy_gboed_freq = accuracy_score(results_['real'].apply(correcao), results_['enriched_gboed_freq'].apply(correcao))
        print("\nACCURACY GBOED FREQ: ",accuracy_gboed_freq,"\n")
        f1_gboed_freq = f1_score(results_['real'].apply(correcao), results_['enriched_gboed_freq'].apply(correcao), average='micro')
        precision_pos_gboed_freq = precision_score(results_['real'].apply(correcao), results_['enriched_gboed_freq'].apply(correcao), pos_label = pos_label, average='macro')
        recall_pos_gboed_freq = recall_score(results_['real'].apply(correcao), results_['enriched_gboed_freq'].apply(correcao), pos_label = pos_label, average='macro')
        precision_neg_gboed_freq = precision_score(results_['real'].apply(correcao), results_['enriched_gboed_freq'].apply(correcao), pos_label = neg_label, average='macro')
        recall_neg_gboed_freq = recall_score(results_['real'].apply(correcao), results_['enriched_gboed_freq'].apply(correcao), pos_label = neg_label, average='macro')
        accuracy_gboed_dist = accuracy_score(results_['real'].apply(correcao), results_['enriched_gboed_dist'].apply(correcao))
        print("\nACCURACY GBOED DIST: ",accuracy_gboed_dist,"\n")
        f1_gboed_dist = f1_score(results_['real'].apply(correcao), results_['enriched_gboed_dist'].apply(correcao), average='micro')
        precision_pos_gboed_dist = precision_score(results_['real'].apply(correcao), results_['enriched_gboed_dist'].apply(correcao), pos_label = pos_label, average='macro')
        recall_pos_gboed_dist = recall_score(results_['real'].apply(correcao), results_['enriched_gboed_dist'].apply(correcao), pos_label = pos_label, average='macro')
        precision_neg_gboed_dist = precision_score(results_['real'].apply(correcao), results_['enriched_gboed_dist'].apply(correcao), pos_label = neg_label, average='macro')
        recall_neg_gboed_dist = recall_score(results_['real'].apply(correcao), results_['enriched_gboed_dist'].apply(correcao), pos_label = neg_label, average='macro')
        # print("\t>RESULTS ENRICHED FREQ\n", results_['enriched_gboed_freq'])
        # print("\t>RESULTS ENRICHED DIST\n", results_['enriched_gboed_dist'])
        th.append({'dataset': dataset,
                   'algorithm': dict_alg[str(alg_)],
                   'threshold': round(threshold_min,2),
                   'accuracy_gboed_freq': accuracy_gboed_freq,
                   'f1_gboed_freq': f1_gboed_freq,
                   'precision_pos_gboed_freq': precision_pos_gboed_freq,
                   'recall_pos_gboed_freq': recall_pos_gboed_freq,
                   'precision_neg_gboed_freq': precision_neg_gboed_freq,
                   'recall_neg_gboed_freq': recall_neg_gboed_freq,
                   'reclass_freq': reclass_freq,
                   'reclass_changed_freq': reclass_changed_freq,
                   'accuracy_gboed_dist': accuracy_gboed_dist,
                   'f1_gboed_dist': f1_gboed_dist,
                   'precision_pos_gboed_dist': precision_pos_gboed_dist,
                   'recall_pos_gboed_dist': recall_pos_gboed_dist,
                   'precision_neg_gboed_dist': precision_neg_gboed_dist,
                   'recall_neg_gboed_dist': recall_neg_gboed_dist,
                   'reclass_dist': reclass_dist,
                   'reclass_changed_dist': reclass_changed_dist,
                   'param1': param1,
                   'param2': param2})
    return pd.DataFrame(th)


def enriquecimento(porcentagem, bow, gboed, th):
    print("DENTRO DA FUNCAO ENRIQUECIMENTO: ", porcentagem, bow, gboed)
    reclass_list = list()
    qtde_reclass = 0
    qtde_reclass_changed = 0
    for p, b, g in zip(porcentagem, bow, gboed):
        # print(p, " ", b, " ", g)
        if(p <= th and gboed != 'neutral'):
            reclass_list.append(g)
            # print(g)
            qtde_reclass = qtde_reclass + 1
            if (g != b):
                qtde_reclass_changed = qtde_reclass_changed + 1
        else:
            reclass_list.append(b)
            # print(b)
    # print("RECLASS: ", reclass_list)
    return reclass_list, qtde_reclass, qtde_reclass_changed


def correcao(x):
    return str(x)


def gboed_save(result_, accuracy_bow, f1_bow, bow_metric, precision_pos_bow, recall_pos_bow, precision_neg_bow, recall_neg_bow):
    
    # print(result_.groupby('threshold')['accuracy_gboed_freq'].mean())
    # print(result_.groupby('threshold')['accuracy_gboed_dist'].mean())
    # print("ACCURACY_GBOED_SINTAX", result_['accuracy_gboed_sintax'])
    
    param1_ = str(result_['param1'].iloc[0])
    param2_ = str(result_['param2'].iloc[0])
    alg = result_['algorithm'].iloc[0]
    if (alg == "KNN"):
        alg_name = alg + '-' + str(param2_)
    elif(alg=="SVM" or alg=="C4.5"):
        alg_name = alg + '-' + str(param1_)
    else:
        alg_name = alg
    output_file = os.path.join(output_path, '2_' + bow_metric + '_' + alg + '_' + str(param2_) + '_' + str(param1_) + '.csv')
    
    with open(output_file, mode='w') as csv_file:
        fieldnames = ['dataset', 
                        'algorithm', 
                        'threshold', 
                        'bow',
                        'f1_bow',
                        'precision_pos_bow',
                        'recall_pos_bow',
                        'precision_neg_bow',
                        'recall_neg_bow',
                        'accuracy_gboed_freq',
                        'f1_gboed_freq',
                        'precision_pos_gboed_freq',
                        'recall_pos_gboed_freq',
                        'precision_neg_gboed_freq',
                        'recall_neg_gboed_freq',
                        'reclass_freq',
                        'reclass_changed_freq',
                        'accuracy_gboed_dist',
                        'f1_gboed_dist',
                        'precision_pos_gboed_dist',
                        'recall_pos_gboed_dist',
                        'precision_neg_gboed_dist',
                        'recall_neg_gboed_dist',
                        'reclass_dist',
                        'reclass_changed_dist',
                        'param1', 
                        'param2']
        writer = csv.DictWriter(csv_file, delimiter=';', dialect='excel', fieldnames=fieldnames)
        writer.writeheader()
        
        for t in range(result_.groupby('threshold').mean().shape[0]):
            reclass_freq_writer = result_.groupby(['threshold','algorithm','param1','param2'])['reclass_freq'].sum().iloc[t]
            reclass_dist_writer = result_.groupby(['threshold','algorithm','param1','param2'])['reclass_dist'].sum().iloc[t]
            print("RECLASS_FREQ: ",reclass_freq_writer)
            print("RECLASS_DIST: ",reclass_dist_writer)
            if reclass_freq_writer != 0 or reclass_dist_writer != 0:
                writer.writerow({'dataset': result_['dataset'].iloc[0],
                                 'algorithm': alg_name,
                                 'threshold': result_.groupby('threshold').mean().index[t],
                                 'bow': np.mean(accuracy_bow),
                                 'f1_bow': np.mean(f1_bow),
                                 'precision_pos_bow': np.mean(precision_pos_bow),
                                 'recall_pos_bow': np.mean(recall_pos_bow),
                                 'precision_neg_bow': np.mean(precision_neg_bow),
                                 'recall_neg_bow': np.mean(recall_neg_bow),
                                 'accuracy_gboed_freq': result_.groupby('threshold').mean()['accuracy_gboed_freq'].iloc[t],
                                 'f1_gboed_freq': result_.groupby('threshold').mean()['f1_gboed_freq'].iloc[t],
                                 'precision_pos_gboed_freq': result_.groupby('threshold').mean()['precision_pos_gboed_freq'].iloc[t],
                                 'recall_pos_gboed_freq': result_.groupby('threshold').mean()['recall_pos_gboed_freq'].iloc[t],
                                 'precision_neg_gboed_freq': result_.groupby('threshold').mean()['precision_neg_gboed_freq'].iloc[t],
                                 'recall_neg_gboed_freq': result_.groupby('threshold').mean()['recall_neg_gboed_freq'].iloc[t],
                                 'reclass_freq': reclass_freq_writer,
                                 'reclass_changed_freq': result_.groupby(['threshold','algorithm','param1','param2'])['reclass_changed_freq'].sum().iloc[t],
                                 'accuracy_gboed_dist': result_.groupby('threshold').mean()['accuracy_gboed_dist'].iloc[t],
                                 'f1_gboed_dist': result_.groupby('threshold').mean()['f1_gboed_dist'].iloc[t],
                                 'precision_pos_gboed_dist': result_.groupby('threshold').mean()['precision_pos_gboed_dist'].iloc[t],
                                 'recall_pos_gboed_dist': result_.groupby('threshold').mean()['recall_pos_gboed_dist'].iloc[t],
                                 'precision_neg_gboed_dist': result_.groupby('threshold').mean()['precision_neg_gboed_dist'].iloc[t],
                                 'recall_neg_gboed_dist': result_.groupby('threshold').mean()['recall_neg_gboed_dist'].iloc[t],
                                 'reclass_dist': reclass_dist_writer,
                                 'reclass_changed_dist': result_.groupby(['threshold','algorithm','param1','param2'])['reclass_changed_dist'].sum().iloc[t],
                                 'param1': param1_,
                                 'param2': param2_})
    print("\tDONE... ", alg, "-", param1_, "-", param2_)



##--------------------------------------------------------------
### MAIN PROGRAM ###

## Executa o treinamento de modelos com os algoritmos KNN, MNB, SVM e C4.5.
## O arquivo de entrada corresponde ao CSV preprocessado e com as predições da gBoED
## geradas pelo script 1_preprocess.py

parser = argparse.ArgumentParser(description='Execute ML algorithms train.')
parser.add_argument('-i', '--input', help='Input preprocessed CSV file.', required=True)
parser.add_argument('-a', '--alg' ,help='0:All algs, 1:KNN, 2:MNB, 3:C4.5, 4:SVM', required=True)
parser.add_argument('-o', '--output_path', help="Output directory.", required=True)
parser.add_argument('-of', '--output_folds_path', help="Output folds directory.", required=True)
parser.add_argument('-d', '--dataset', help="Dataset.", required=True)
parser.add_argument('-pos', '--pos_label', help="Set the positive class.", required=False, default="positive")
parser.add_argument('-neg', '--neg_label', help="Set the negative class.", required=False, default="negative")
parser.add_argument('-p', '--procs', help="Number of processors. If 0 all processors will be used.", required=False, default="0")
args = parser.parse_args()

input_ = args.input
alg = int(args.alg)
dict_alg = {'1':'KNN', '2':'MNB', '3':'C4.5', '4':'SVM'}
output_path = args.output_path
output_folds_path = args.output_folds_path
dataset = args.dataset
pos_label = args.pos_label
neg_label = args.neg_label
procs = int(args.procs)
if int(args.procs) == 0:
    procs = mp.cpu_count()
else:
    procs = int(args.procs)

#print(input_, alg, output_path, dataset, procs)
df_input = pd.read_csv(input_, sep='|')
#df_input = df_input.sample(100)
#print(df_input)
data = df_input['bow_preproc'].apply(lambda x: np.str_(x))
labels = df_input['classes']
filenames = df_input['file_name']
#print("Database: ", np.shape(X), np.unique(y, return_counts=True))

# K Fold não aleatório
#kf = KFold(n_splits=10, shuffle = False)
# K Fold aleatório estratificado
# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
# K Fold aleatório nao-estratificado
kf = KFold(n_splits=10, shuffle=True, random_state=1234)
p = Pool(processes=procs)


if (alg == 0):
    start_job(1, "tf", dataset, pos_label, neg_label)
    start_job(2, "tf", dataset, pos_label, neg_label)
    start_job(3, "tf", dataset, pos_label, neg_label)
    start_job(4, "tf", dataset, pos_label, neg_label)
    start_job(1, "tfidf", dataset, pos_label, neg_label)
    start_job(2, "tfidf", dataset, pos_label, neg_label)
    start_job(3, "tfidf", dataset, pos_label, neg_label)
    start_job(4, "tfidf", dataset, pos_label, neg_label)

else:
    start_job(alg, "tf", dataset, pos_label, neg_label)
    start_job(alg, "tfidf", dataset, pos_label, neg_label)

p.close()
p.join()