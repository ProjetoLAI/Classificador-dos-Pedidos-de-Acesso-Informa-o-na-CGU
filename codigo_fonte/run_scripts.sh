### =========== ###
### PRE PROCESS ###
### =========== ###

### CGU_pedidos_inf_2020_balanceado_sem_genericos ###
nohup python3 1_preprocess.py -i /media/hd2/databases/CGU_pedidos_inf_2020/cgu_pedidos_inf_concedido_negado_balanceado_sem_genericos/ -o results_cgu_pedidos_inf_2020_balanceado_sem_genericos/ -l portuguese -d CGU_pedidos_inf_2020_balanceado_sem_genericos -s results_cgu_pedidos_inf_2020_balanceado_sem_genericos/0_lists/stopPort.txt -dt results_cgu_pedidos_inf_2020_balanceado_sem_genericos/0_lists/domain_terms_cgu_ped_inf_2020_perguntas.csv -pl results_cgu_pedidos_inf_2020_balanceado_sem_genericos/0_lists/concedidos_cgu_ped_inf_2020_perguntas.csv -nl results_cgu_pedidos_inf_2020_balanceado_sem_genericos/0_lists/negados_cgu_ped_inf_2020_perguntas.csv -pc concedido -nc negado -nnc neutro > results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_log_preprocess_cgu_pedidos_inf_2020.txt &

### ====================== ###
### PRE PROCESS STATISTICS ###
### ====================== ###

### CGU_pedidos_inf_2020_balanceado_sem_genericos ###
python3 analyse_results_preprocess_freq_dist.py -ir results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_representations/1_gboed_freq_cgu_pedidos_inf_2020_balanceado_sem_genericos.csv -is results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_preprocess_cgu_pedidos_inf_2020_balanceado_sem_genericos.csv -sc gboed_freq_class -m freq -l portuguese > results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_log_analyse_results_gboed_freq_cgu_pedidos_inf_2020.txt
python3 analyse_results_preprocess_freq_dist.py -ir results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_representations/1_gboed_dist_cgu_pedidos_inf_2020_balanceado_sem_genericos.csv -is results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_preprocess_cgu_pedidos_inf_2020_balanceado_sem_genericos.csv -sc gboed_dist_class -m dist -l portuguese > results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_log_analyse_results_gboed_dist_cgu_pedidos_inf_2020.txt

### ======================== ###
### CLEANNING CLASSIFICATION ###
### ======================== ###

### CGU_pedidos_inf_2020_balanceado_sem_genericos ###
python3 0_clean_folds.py -cl False -cr False -cc True -p results_cgu_pedidos_inf_2020_balanceado_sem_genericos

### ============== ###
### CLASSIFICATION ###
### ============== ###

### CGU_pedidos_inf_2020_balanceado_sem_genericos ###
nohup python3 2_classify_multiprocess_all_algs.py -a 0 -i results_cgu_pedidos_inf_2020_balanceado_sem_genericos/1_preprocess_cgu_pedidos_inf_2020_balanceado_sem_genericos.csv -o results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_partial_results/ -of results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_partial_results/folds/ -d CGU_pedidos_inf_2020_balanceado_sem_genericos -pos concedido -neg negado -p 2 > results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_classify_multiprocess_all_algs.txt &

### ========================= ###
### CLASSIFICATION STATISTICS ###
### ========================= ###

### CGU_pedidos_inf_2020_balanceado_sem_genericos ###
python3 3_concat_and_normalize_classification_results.py -i results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_partial_results/ -otf results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_tf_final_results.csv -otfidf results_cgu_pedidos_inf_2020_balanceado_sem_genericos/2_tfidf_final_results.csv -final False
