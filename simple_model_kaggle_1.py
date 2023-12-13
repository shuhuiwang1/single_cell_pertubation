# test on using mean and std as features 
from scripts.simple_model import simple_model_feature
from scripts.data import CustomDataset_1
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
def train_model(model: simple_model_feature, data_train: CustomDataset_1, batch_size = 2, max_epoch = 10, data_test = None):
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
                genes, drugs, genes_reduce, cells = data[0], data[1], data[2], data[3]
                train_error = model.update(
                    drugs, cells, genes_reduce 
                )
                train_loss += [train_error]
            r2_loss = []
            if data_test != None:
                model.eval()
                test_loss = []
                loss_18_l = []
                with torch.no_grad():
                    for data in datasets_test:
                        genes, drugs, genes_reduce, cells = data[0], data[1], data[2], data[3]
                        gene_reconstructions = model.predict(
                            drugs, cells
                        )
                        genes_predict = data_train.reconstruct(gene_reconstructions)
                        reconstruction_loss = model.loss_autoencoder(gene_reconstructions, genes_reduce.to(model.device))
                        test_loss += [reconstruction_loss.item()]
                        y_true = genes.cpu().detach().numpy()
                        y_pred = genes_predict
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
        tsave(best_model.state_dict(), 'pretrain/simple_kaggle.pt')
        print(best_acc)
    tsave(best_model.state_dict(), 'pretrain/simple_kaggle.pt')
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

def predict(features_dict, data_train):
    path = '/home/alexandre/Downloads/single_cell_pertubation_kaggle/'
    submit = pd.read_csv(path + 'kaggle/input/sample_submission.csv')
    submit_id = pd.read_csv(path + 'kaggle/input/id_map.csv')
    path1 = '/home/alexandre/Downloads/single_cell_pertubation_kaggle/'
    num = len(submit_id)
    gene_all = []
    for id in tqdm(range(num)):
        cell_name = submit_id['cell_type'].iloc[id]
        drug_name = submit_id['sm_name'].iloc[id]
        drug_feature = features_dict[drug_name]
        cell_feature = features_dict[cell_name]
        drugs = torch.tensor(np.vstack((drug_feature, drug_feature))).to(dtype=torch.float32)
        cells = torch.tensor(np.vstack((cell_feature, cell_feature))).to(dtype=torch.float32)
        # print(drug_id, cell_id)
        gene_reconstructions = model.predict(
                            drugs, cells
                        )
        gene_reconstruct = data_train.reconstruct(gene_reconstructions)
        # print(gene_reconstruct.mean(axis = 0))
        gene_all.append([gene_reconstruct.mean(axis = 0)])
    Y_submit_pred = np.concatenate(gene_all) 
    return  Y_submit_pred
if __name__ == '__main__':
    # load data 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) 
    adata_de = sc.read('../data/kaggle_train_de.h5ad')
    # features_dict = joblib.load('pretrain/features_dict.joblib')
    features_dict = joblib.load('pretrain/features_pca_dict.joblib')
    n_comp = 50 
    dataset = CustomDataset_1(adata_de, n_comp, feature_dict=features_dict)
    folds_index_data_AmbrosM = fold_design(adata_de)
    results = {}
    num = 255 
    Y_submit_pred = np.zeros((num,  18211 )); i_blend_for_submit = 0
    for fold in list(folds_index_data_AmbrosM.keys()):
        id_test = folds_index_data_AmbrosM[fold]['test']
        id_train = folds_index_data_AmbrosM[fold]['train']
        
        dataset_train = CustomDataset_1(adata_de[id_train], n_comp=n_comp, feature_dict=features_dict, pca_model=None) 
        dataset_test =  CustomDataset_1(adata_de[id_test], n_comp=n_comp, feature_dict=features_dict, pca_model=dataset_train.pca_model) 
        hparam = {
        'hid_dim': 2,
        'mlp_size': [2**7, 2**6],
        'dim_out': n_comp,
        'device': device,
        'lr': 1e-3,
        'wd': 1e-7,
        'n_feature': 50}
        model = simple_model_feature(hparam=hparam)
        model = model.to(model.device)
        model, best_acc = train_model(model, dataset_train, batch_size= 100, max_epoch = 200, data_test= dataset_test)
        results[fold] = best_acc
        Y_submit_pred_from_one_fold = predict(features_dict = features_dict, data_train=dataset_train)
        Y_submit_pred =  (Y_submit_pred * i_blend_for_submit + Y_submit_pred_from_one_fold )/(i_blend_for_submit + 1)
        i_blend_for_submit += 1
    rmse = []
    for key in results:
        rmse += [results[key][0]]
    print(np.mean(rmse))
    path = '/home/alexandre/Downloads/single_cell_pertubation_kaggle/'
    submit = pd.read_csv(path + 'kaggle/input/sample_submission.csv')
    df_to_submit = pd.DataFrame(Y_submit_pred, columns=list(submit.columns[1:]), index= list(submit.index))
    df_to_submit.index.name = ('id')
    print(df_to_submit.head(3))
    df_to_submit.to_csv('results/MLP_50_cell_drugs_mean_std_pca.csv')


# !kaggle competitions submit -c open-problems-single-cell-perturbations -f results/MLP_50_cell_drugs_mean_std_pca.csv -m 'mlp model with mean and std'
