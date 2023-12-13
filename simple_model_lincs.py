# test on using mean and std as features 
from scripts.simple_model import simple_model_feature
from scripts.data import CustomDataset_lincs
import torch 
from torch import save as tsave 
import numpy as np 
from scipy.stats import pearsonr 
import scanpy as sc 
import joblib 
import pandas as pd 
from tqdm import tqdm 
def mrrmse_nump(y_real, y_pred):
    return np.mean(np.sqrt(np.mean((y_real - y_pred)**2, axis = 1)))
def train_model(model: simple_model_feature, data_train: CustomDataset_lincs, batch_size = 2, max_epoch = 10, data_test = None):
    datasets_train = torch.utils.data.DataLoader(
                    data_train, batch_size=batch_size, shuffle=True
                )
    best_val = 10000 
    if data_test != None:
        datasets_test = torch.utils.data.DataLoader(
                    data_test, batch_size=batch_size, shuffle=True
                )
    try:
        for epoch in range(max_epoch):
            train_loss = []
            model.train()
            for id, data in enumerate(datasets_train): 
                genes, drugs, cells = data[0], data[1], data[2]
                train_error = model.update(
                    drugs, cells, genes 
                )
                train_loss += [train_error]
            r2_loss = []
            if data_test != None:
                model.eval()
                test_loss = []
                loss_18_l = []
                with torch.no_grad():
                    for data in datasets_test:
                        genes, drugs, cells = data[0], data[1], data[2]
                        gene_reconstructions = model.predict(
                            drugs, cells
                        )
                        reconstruction_loss = model.loss_autoencoder(gene_reconstructions, genes.to(model.device))
                        test_loss += [reconstruction_loss.item()]
                        y_true = genes.cpu().detach().numpy()
                        y_pred = gene_reconstructions.cpu().detach().numpy()
                        r2_loss += [pearsonr(y_true[i,:], y_pred[i,:])[0] for i in range(len(genes))]
                        loss_18_l +=  [mrrmse_nump(y_true, y_pred)]
                if epoch % 100 == 0:
                    print(f'epoch: {epoch} with loss: {np.mean(train_loss)}') 
                    print(f'epoch: {epoch} with validation loss: {np.mean(test_loss)} on 18k genes {np.mean(loss_18_l)} with r2: {np.mean(r2_loss)}')

                if np.mean(loss_18_l) < best_val:
                    best_val = np.mean(loss_18_l)
                    best_model = model 
                    best_acc = [np.mean(loss_18_l), np.mean(train_loss), np.mean(r2_loss)]
    except KeyboardInterrupt:
        print('save model by keyboard interruption')
        tsave(best_model.state_dict(), 'pretrain/simple_lincs.pt')
        print(best_acc)
    tsave(best_model.state_dict(), 'pretrain/simple_lincs.pt')
    print(best_acc)
    return best_model, best_acc

def fold_design(adata_de):
    folds_index_data_AmbrosM = {}
    train_sm_names = ['Idelalisib', 'Crizotinib', 'Linagliptin', 'Palbociclib', 'Dabrafenib', 'Alvocidib', 'LDN 193189', 'R428', 'Porcn Inhibitor III', 
    'Belinostat', 'Foretinib', 'MLN 2238', 'Penfluridol', 'Dactolisib', 'O-Demethylated Adapalene', 'Oprozomib (ONX 0912)', 'CHIR-99021']
   
    list_fold_ids =  ['NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells']
    for fold_id in list_fold_ids:
    
            mask_va = (adata_de.obs.cell_type == fold_id) & (~adata_de.obs.sm_name.isin(train_sm_names))
            mask_tr =(adata_de.obs.cell_type != fold_id) | adata_de.obs.sm_name.isin(train_sm_names)
        
            IX_train = np.where( mask_tr > 0 )[0]
            IX_test = np.where( mask_va > 0 )[0]
            folds_index_data_AmbrosM[fold_id] = {
                'train': IX_train,
                'test': IX_test 
            }
    return folds_index_data_AmbrosM

if __name__ == '__main__':
    # load data 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) 
    # adata_de = sc.read('/home/alexandre/Downloads/single_cell_pertubation_kaggle/DE_multistep_model/code_and_data_v2/pretrain_data/DE_lincs_kaggle_all.h5ad')
    adata_de = sc.read('../data/DE_lincs_kaggle_all.h5ad')
    adata_de = adata_de[adata_de.obs.study == 'lincs']
    features_dict = joblib.load('pretrain/features_dict_lincs.joblib')
    adata_train = adata_de[adata_de.obs.split == 'train']
    adata_test = adata_de[adata_de.obs.split == 'test']
    
    dataset_train = CustomDataset_lincs(adata_train, feature_dict=features_dict) 
    dataset_test =  CustomDataset_lincs(adata_test, feature_dict=features_dict) 
    hparam = {
    'hid_dim': 2,
    'mlp_size': [2**7, 2**6],
    'dim_out': 918,
    'device': device,
    'lr': 1e-3,
    'wd': 1e-7,
    'n_feature': 918}
    model = simple_model_feature(hparam=hparam)
    model = model.to(model.device)
    model, best_acc = train_model(model, dataset_train, batch_size= 100, max_epoch = 300, data_test= dataset_test)
       
