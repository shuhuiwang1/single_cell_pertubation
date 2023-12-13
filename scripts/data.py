
from torch.utils.data import Dataset 
import torch 
from sklearn.preprocessing import OneHotEncoder 
import numpy as np 
from sklearn.decomposition import PCA

class CustomDataset(Dataset):
    def __init__(self, adata_de, n_comp = 50, encoder = None, pca_model = None):
        self.genes = adata_de.X
        if encoder == None:
            self.encoder_drug = OneHotEncoder(sparse=False)
            drugs_unique = np.array(adata_de.obs.sm_name.unique()) 
            cell_unique =  np.array(adata_de.obs.cell_type.unique()) 
            self.encoder_drug.fit(drugs_unique.reshape(-1,1))
            self.encoder_cell = OneHotEncoder(sparse=False)
            self.encoder_cell.fit(cell_unique.reshape(-1,1))
        else:
            self.encoder_drug = encoder.encoder_drug
            self.encoder_cell = encoder.encoder_cell  
        if pca_model == None:
            self.pca_model = PCA(n_components=n_comp) 
            self.pca_model.fit(self.genes)
        else:
            self.pca_model = pca_model
        
        self.genes_reduce = self.pca_model.transform(self.genes)


        drug_names = np.array(adata_de.obs['sm_name'].values)
        cell_names = np.array(adata_de.obs['cell_type'].values)
        drugs_ohe = self.encoder_drug.transform(drug_names.reshape(-1,1))
        cell_ohe = self.encoder_cell.transform(cell_names.reshape(-1,1))
        self.drugs = torch.tensor(drugs_ohe.argmax(axis = 1).reshape(-1,1))
        self.cells = torch.tensor(cell_ohe.argmax(axis = 1).reshape(-1,1))
        

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        return (
            self.genes[idx],
            self.drugs[idx],
            self.genes_reduce[idx],
            self.cells[idx]) 
    
    def reconstruct(self, genes_proj):
        if torch.is_tensor(genes_proj):
            genes_proj = genes_proj.cpu().detach().numpy()
        recons = self.pca_model.inverse_transform(genes_proj)
        return recons




class CustomDataset_1(Dataset):
    def __init__(self, adata_de, n_comp = 50, feature_dict = None, pca_model = None):
        self.genes = adata_de.X
        if pca_model == None:
            self.pca_model = PCA(n_components=n_comp) 
            self.pca_model.fit(self.genes)
        else:
            self.pca_model = pca_model
        
        self.genes_reduce = self.pca_model.transform(self.genes)


        drug_names = np.array(adata_de.obs['sm_name'].values)
        cell_names = np.array(adata_de.obs['cell_type'].values)
        drugs_features = []
        for drug in drug_names:
            drugs_features +=  [[feature_dict[drug].T]]
        cells_features = []
        for cell in cell_names:
            cells_features +=  [[feature_dict[cell].T]]

        self.drugs = torch.tensor(np.concatenate(drugs_features)).to(dtype=torch.float32)
        self.cells = torch.tensor(np.concatenate(cells_features)).to(dtype=torch.float32)
        

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        return (
            self.genes[idx],
            self.drugs[idx],
            self.genes_reduce[idx],
            self.cells[idx]) 
    
    def reconstruct(self, genes_proj):
        if torch.is_tensor(genes_proj):
            genes_proj = genes_proj.cpu().detach().numpy()
        recons = self.pca_model.inverse_transform(genes_proj)
        return recons
    

from numpy import logspace
def design_dose_vec(num_step, dose_max):
    dose_vec = []
    for id in np.arange(num_step):
        el = np.exp(np.log(dose_max) - id*np.log(3)/3)
        dose_vec.append(round(el,2))
    dose_vec = dose_vec[::-1]
    dose_vec[0] = 0 
    return dose_vec

class CustomDataset_2(Dataset):
    def __init__(self, adata_de, n_comp = 50, n_step = 5, feature_dict = None, pca_model = None):
        self.genes = torch.tensor(np.zeros((len(adata_de), 918)))
        self.genes_18 = adata_de.X
        
        if pca_model == None:
            self.pca_model = PCA(n_components=n_comp) 
            self.pca_model.fit(self.genes_18)
        else:
            self.pca_model = pca_model
        
        self.genes_reduce = self.pca_model.transform(self.genes_18)
        dose_l = design_dose_vec(n_step, 1)
        self.drugs_dose = torch.tensor([dose_l for i in range(len(self.genes))])

        drug_names = np.array(adata_de.obs['sm_name'].values)
        cell_names = np.array(adata_de.obs['cell_type'].values)
        drugs_features = []
        for drug in drug_names:
            drugs_features +=  [[feature_dict[drug].T]]
        cells_features = []
        for cell in cell_names:
            cells_features +=  [[feature_dict[cell].T]]

        self.drugs = torch.tensor(np.concatenate(drugs_features)).to(dtype=torch.float32)
        self.cells = torch.tensor(np.concatenate(cells_features)).to(dtype=torch.float32)
        self.genes_18 = torch.tensor(self.genes_18).to(dtype=torch.float32)
        self.genes = torch.tensor(self.genes).to(dtype=torch.float32)
        self.genes_reduce = torch.tensor(self.genes_reduce).to(dtype=torch.float32)
        

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        return (
            self.genes[idx],
            self.drugs[idx],
            self.genes_reduce[idx],
            self.cells[idx],
            self.drugs_dose[idx], 
            self.genes_18[idx]) 
    
    def reconstruct(self, genes_proj):
        if torch.is_tensor(genes_proj):
            genes_proj = genes_proj.cpu().detach().numpy()
        recons = self.pca_model.inverse_transform(genes_proj)
        return recons



class CustomDataset_3(Dataset):
    def __init__(self, adata_de, n_comp = 50, n_step = 5, encoder = None, pca_model = None, feature_dict = None):
        self.genes = torch.tensor(np.zeros((len(adata_de), 918))).to(dtype=torch.float32)
        genes_18 = adata_de.X
        dose_l = design_dose_vec(n_step, 1)
        self.drugs_dose = torch.tensor([dose_l for i in range(len(adata_de))])
        if encoder == None:
            self.encoder_drug = OneHotEncoder(sparse=False)
            drugs_unique = np.array(adata_de.obs.sm_name.unique()) 
            cell_unique =  np.array(adata_de.obs.cell_type.unique()) 
            self.encoder_drug.fit(drugs_unique.reshape(-1,1))
            self.encoder_cell = OneHotEncoder(sparse=False)
            self.encoder_cell.fit(cell_unique.reshape(-1,1))
        else:
            self.encoder_drug = encoder.encoder_drug
            self.encoder_cell = encoder.encoder_cell  
        if pca_model == None:
            self.pca_model = PCA(n_components=n_comp) 
            self.pca_model.fit(genes_18)
        else:
            self.pca_model = pca_model
        
        genes_reduce = self.pca_model.transform(genes_18)


        drug_names = np.array(adata_de.obs['sm_name'].values)
        cell_names = np.array(adata_de.obs['cell_type'].values)
        drugs_ohe = self.encoder_drug.transform(drug_names.reshape(-1,1))
        cell_ohe = self.encoder_cell.transform(cell_names.reshape(-1,1))
        self.drugs = torch.tensor(drugs_ohe.argmax(axis = 1).reshape(-1,1))
        self.cells = torch.tensor(cell_ohe.argmax(axis = 1).reshape(-1,1))

        self.genes_18 = torch.tensor(genes_18).to(dtype=torch.float32)
        self.genes_reduce = torch.tensor(genes_reduce).to(dtype=torch.float32)
        self.feature = False
        if feature_dict != None:
            self.feature = True
            drugs_features = []
            for drug in drug_names:
                drugs_features +=  [[feature_dict[drug].T]]
            cells_features = []
            for cell in cell_names:
                cells_features +=  [[feature_dict[cell].T]]

            self.drugs_feature = torch.tensor(np.concatenate(drugs_features)).to(dtype=torch.float32)
            self.cells_feature = torch.tensor(np.concatenate(cells_features)).to(dtype=torch.float32)
            

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        if self.feature:
            return (
                self.genes[idx],
                self.drugs[idx],
                self.genes_reduce[idx],
                self.cells[idx],
                self.drugs_dose[idx], 
                self.genes_18[idx],
                self.drugs_feature[idx],
                self.cells_feature[idx]) 
        else:
            return (
                self.genes[idx],
                self.drugs[idx],
                self.genes_reduce[idx],
                self.cells[idx],
                self.drugs_dose[idx], 
                self.genes_18[idx]) 

    
    def reconstruct(self, genes_proj):
        if torch.is_tensor(genes_proj):
            genes_proj = genes_proj.cpu().detach().numpy()
        recons = self.pca_model.inverse_transform(genes_proj)
        return recons
    





class CustomDataset_lincs(Dataset):
    def __init__(self, adata_de, feature_dict = None):
        self.genes = torch.tensor(adata_de.X)

        drug_names = np.array(adata_de.obs['pert_iname'].values)
        cell_names = np.array(adata_de.obs['cell_id'].values)
        drugs_features = []
        for drug in drug_names:
            drugs_features +=  [[feature_dict[drug].T]]
        cells_features = []
        for cell in cell_names:
            cells_features +=  [[feature_dict[cell].T]]

        self.drugs = torch.tensor(np.concatenate(drugs_features)).to(dtype=torch.float32)
        self.cells = torch.tensor(np.concatenate(cells_features)).to(dtype=torch.float32)
        

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        return (
            self.genes[idx],
            self.drugs[idx],
            self.cells[idx]) 



class CustomDataset_lincs_msd(Dataset):
    def __init__(self, adata_de, encoder = None, feature_dict = None):
        self.genes = torch.tensor(adata_de.X[:, :918]).to(dtype=torch.float32)
        self.genes_aft = torch.tensor(adata_de.X[:, 918:]).to(dtype=torch.float32)
        self.drugs_dose = torch.tensor(adata_de.obsm['dose']).to(dtype=torch.float32)
        if encoder == None:
            self.encoder_drug = OneHotEncoder(sparse=False)
            drugs_unique = np.array(adata_de.obs.pert_iname.unique()) 
            cell_unique =  np.array(adata_de.obs.cell_id.unique()) 
            self.encoder_drug.fit(drugs_unique.reshape(-1,1))
            self.encoder_cell = OneHotEncoder(sparse=False)
            self.encoder_cell.fit(cell_unique.reshape(-1,1))
        else:
            self.encoder_drug = encoder.encoder_drug
            self.encoder_cell = encoder.encoder_cell  


        drug_names = np.array(adata_de.obs['pert_iname'].values)
        cell_names = np.array(adata_de.obs['cell_id'].values)
        drugs_ohe = self.encoder_drug.transform(drug_names.reshape(-1,1))
        cell_ohe = self.encoder_cell.transform(cell_names.reshape(-1,1))
        self.drugs = torch.tensor(drugs_ohe.argmax(axis = 1).reshape(-1,1))
        self.cells = torch.tensor(cell_ohe.argmax(axis = 1).reshape(-1,1))


        self.feature = False
        if feature_dict != None:
            self.feature = True
            drugs_features = []
            for drug in drug_names:
                drugs_features +=  [[feature_dict[drug].T]]
            cells_features = []
            for cell in cell_names:
                cells_features +=  [[feature_dict[cell].T]]

            self.drugs_feature = torch.tensor(np.concatenate(drugs_features)).to(dtype=torch.float32)
            self.cells_feature = torch.tensor(np.concatenate(cells_features)).to(dtype=torch.float32)
            

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        if self.feature:
            return (
                self.genes[idx],
                self.drugs[idx],
                self.cells[idx],
                self.drugs_dose[idx], 
                self.genes_aft[idx],
                self.drugs_feature[idx],
                self.cells_feature[idx]) 
        else:
            return (
                self.genes[idx],
                self.drugs[idx],
                self.cells[idx],
                self.drugs_dose[idx], 
                self.genes_aft[idx]) 


    