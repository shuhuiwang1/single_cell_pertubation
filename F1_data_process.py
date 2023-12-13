# get drug and cell mean, std as features 

import scanpy as sc 
import numpy as np 
adata_de = sc.read('../data/kaggle_train_de.h5ad') 
features_dic = {}
for cell in adata_de.obs.cell_type.unique():
    adata_sub = adata_de[adata_de.obs.cell_type == cell]
    features_dic[cell] = np.hstack((np.mean(adata_sub.X, axis=0), np.std(adata_sub.X, axis = 0)))

for drug in adata_de.obs.sm_name.unique():
    adata_sub = adata_de[adata_de.obs.sm_name == drug]
    features_dic[drug] = np.hstack((np.mean(adata_sub.X, axis=0), np.std(adata_sub.X, axis = 0)))


import joblib
joblib.dump(features_dic, 'pretrain/features_dict.joblib')


# get features for lincs data 
import scanpy as sc 
import joblib 
import numpy as np 
adata_de = sc.read('../data/DE_lincs_kaggle_all.h5ad')
adata_de = adata_de[adata_de.obs.study == 'lincs']

features_dic = {}
for cell in adata_de.obs.cell_id.unique():
    adata_sub = adata_de[adata_de.obs.cell_id == cell]
    features_dic[cell] = np.hstack((np.mean(adata_sub.X, axis=0), np.std(adata_sub.X, axis = 0)))

for drug in adata_de.obs.pert_iname.unique():
    adata_sub = adata_de[adata_de.obs.pert_iname == drug]
    features_dic[drug] = np.hstack((np.mean(adata_sub.X, axis=0), np.std(adata_sub.X, axis = 0)))


import joblib
joblib.dump(features_dic, 'pretrain/features_dict_lincs.joblib')