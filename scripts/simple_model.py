from torch import nn, cat as tcat, sqrt, optim, dtype

class rrmse(nn.Module):
    def __init__(self):
        super(rrmse, self).__init__()
    
    def forward(self, y_real, y_pred):
        return sqrt((((y_real - y_pred)**2).mean(axis = 1))).mean()

class MLP(nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, bias_f = True, drop = True):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[s], sizes[s + 1], bias=bias_f),
                nn.Dropout(0.1) if drop else None,
                nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-2]
        self.relu = nn.ReLU() 
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)   


class simple_model(nn.Module):
    def __init__(self, hparam):
        super(simple_model, self).__init__()
        self.hparam = hparam 
        self.device = self.hparam['device']
        dim_hid = hparam['hid_dim']
        self.cell_embeddings = nn.Embedding(6, dim_hid)
        self.drug_embeddings = nn.Embedding(146, dim_hid)
        self.fit_pred = MLP(
            [dim_hid*2]
            + hparam['mlp_size']
            + [hparam['dim_out']],
        )
        self.cell_weight = torch.nn.Parameter(torch.zeros(2,4))
        self.loss_autoencoder = rrmse()
        self.optimizer_autoencoder = optim.Adam(self.parameters(), lr = self.hparam['lr'], weight_decay = self.hparam['wd'])
        self.cell_l_4  = hparam['cell_l_4']
        self.cell_l_2 =  hparam['cell_l_2']

    
    def compute_cell_embeding(self, cells):
        # 'NK cells': 28, 'T cells CD4+': 29, 'T cells CD8+': 30, 'T regulatory cells': 31, 'B cells': 32, 'Myeloid cells': 33
        self.pretrain_cell_emb = self.cell_embeddings.weight[self.cell_l_4,:]
        cell_weight = []
        for cell in cells:
            if cell in self.cell_l_2:
                id = self.cell_l_2.index(cell)
                cell_weight.append((self.cell_weight[id]@self.cell_embeddings.weight[self.cell_l_4,:]).view(1,-1))
            else:
                cell_weight.append(( self.cell_embeddings(cell)).view(1,-1))
        return torch.cat(cell_weight, 0)
    def predict(
        self, 
        drugs, 
        cells, 
    ):

        drugs, cells = drugs.to(self.device), cells.to(self.device)
        emb_1, emb_2 = self.cell_embeddings(cells).squeeze(1), self.drug_embeddings(drugs).squeeze(1)
        # emb_1 = self.compute_cell_embeding(cells) 
        # emb_2 = self.drug_embeddings(drugs).squeeze(1)
        emb = tcat((emb_1, emb_2), dim=1)
        gene_reconstructions = self.fit_pred(emb)
        return gene_reconstructions



    def update(self, drugs, cells, genes_d):
        drugs, cells = drugs.to(self.device), cells.to(self.device)
        gene_reconstructions = self.predict(drugs, cells)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes_d)
    
        self.optimizer_autoencoder.zero_grad()

        reconstruction_loss.backward()
        self.optimizer_autoencoder.step()
       
        return reconstruction_loss.item() 
    


class simple_model_feature(nn.Module):
    def __init__(self, hparam):
        super(simple_model_feature, self).__init__()
        self.hparam = hparam 
        self.device = self.hparam['device']
        dim_hid = hparam['hid_dim']
        self.cell_embeddings = nn.Linear(hparam['n_feature']*2, dim_hid)
        self.drug_embeddings = nn.Linear(hparam['n_feature']*2, dim_hid)
        self.fit_pred = MLP(
            [dim_hid*2]
            + hparam['mlp_size']
            + [hparam['dim_out']],
        )
        self.loss_autoencoder = rrmse()
        self.optimizer_autoencoder = optim.Adam(self.parameters(), lr = self.hparam['lr'], weight_decay = self.hparam['wd'])
    
    def predict(
        self, 
        drugs, 
        cells, 
    ):
        
        drugs, cells = drugs.to(self.device), cells.to(self.device)
        emb_1, emb_2 = self.cell_embeddings(cells), self.drug_embeddings(drugs)
        emb = tcat((emb_1, emb_2), dim=1)
        gene_reconstructions = self.fit_pred(emb)
        return gene_reconstructions



    def update(self, drugs, cells, genes_d):
        drugs, cells = drugs.to(self.device), cells.to(self.device)
        gene_reconstructions = self.predict(drugs, cells)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes_d)
    
        self.optimizer_autoencoder.zero_grad()

        reconstruction_loss.backward()
        self.optimizer_autoencoder.step()
       
        return reconstruction_loss.item() 





import torch 
from torch import cat as tcat, nn 

class rrmse(torch.nn.Module):
    def __init__(self):
        super(rrmse, self).__init__()

    def forward(self, y_real, y_pred, sigma=None):
        # y_real_ = torch.clamp(y_real, min = -50, max = 50)
        if sigma is None:
            return torch.sqrt((((y_real - y_pred)**2).mean(axis = 1))).mean()
        else:
            return torch.sqrt((((y_real - y_pred)**2 * sigma).mean(axis = 1))).mean()
class Basic_ff(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, in_dim, out_dim, hid, bias_f = True, last_layer = 'linear'):
        super(Basic_ff, self).__init__()
        self.f1 = torch.nn.Linear(in_dim, hid)
        self.f2 = torch.nn.Linear(hid, out_dim)
        self.rel = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(hid)
        self.out = torch.nn.Dropout(0.1)

    def forward(self, x, prev=None):
        x_ = tcat([x, prev], dim=1)
        res = self.rel(self.bn(self.f1(x_)))
        res = x + self.f2(res)
        return res


class Simple_residual(torch.nn.Module):
    def __init__(self, hparam):
        super(Simple_residual, self).__init__()

        self.loss_autoencoder = rrmse()
        self.device = hparam['device']
        # simple modifs
        self.drug_embeddings = torch.nn.Embedding(hparam["nb_drug"], hparam["dim_emb"])
        self.cell_embeddings = torch.nn.Embedding(hparam["nb_cell"], hparam["dim_emb"])
        self.pred_genes = torch.nn.ModuleList()
        self.convert_in = torch.nn.Linear(hparam["dim_emb"]*2, hparam["dim_emb"])
        for i in range(hparam["nb_layer"]-1):
            self.pred_genes += [Basic_ff(hparam["dim_emb"]*3, hparam["dim_emb"], hparam["hid"])]
        self.convert_out = torch.nn.Linear(hparam["dim_emb"], hparam["nb_feat"])
        # self.convert_out1 = torch.nn.Linear(hparam["dim_emb"], 918)
        # self.convert_out2 = torch.nn.Linear(918, hparam["nb_feat"])
        self.optimizer_autoencoder = torch.optim.Adam(self.parameters(), lr = hparam['lr'], weight_decay = hparam['wd'])

    def predict(self, drugs, cells):
        latent_drug = self.drug_embeddings(drugs)
        latent_cell = self.cell_embeddings(cells)
        lat_comp = tcat([latent_drug.squeeze(1), latent_cell.squeeze(1)], dim=1)
        emb = self.convert_in(lat_comp)
        for layer in self.pred_genes:
            emb = layer(emb, lat_comp)
        gene_reconstructions = self.convert_out(emb)
        return gene_reconstructions

    def update(self, drugs, cells, genes):
        gene_reconstructions  = self.predict(drugs, cells)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)

        self.optimizer_autoencoder.zero_grad()
        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2)
        self.optimizer_autoencoder.step()

        return reconstruction_loss.item()
    

class Simple_residual_feature(torch.nn.Module):
    def __init__(self, hparam):
        super(Simple_residual_feature, self).__init__()

        self.loss_autoencoder = rrmse()
        self.device = hparam['device']
        # simple modifs
        self.cell_embeddings = nn.Linear(hparam['n_feature']*2,  hparam["dim_emb"])
        self.drug_embeddings = nn.Linear(hparam['n_feature']*2,  hparam["dim_emb"])
        self.pred_genes = torch.nn.ModuleList()
        self.convert_in = torch.nn.Linear(hparam["dim_emb"]*2, hparam["dim_emb"])
        for i in range(hparam["nb_layer"]-1):
            self.pred_genes += [Basic_ff(hparam["dim_emb"]*3, hparam["dim_emb"], hparam["hid"])]
        # self.convert_out = torch.nn.Linear(hparam["dim_emb"], hparam["nb_feat"])
        self.convert_out = torch.nn.Linear(hparam["dim_emb"], hparam["nb_feat"])
        self.optimizer_autoencoder = torch.optim.Adam(self.parameters(), lr = hparam['lr'], weight_decay = hparam['wd'])

    def predict(self, drugs, cells):
        latent_drug = self.drug_embeddings(drugs)
        latent_cell = self.cell_embeddings(cells)
        lat_comp = tcat([latent_drug, latent_cell], dim=1)
        emb = self.convert_in(lat_comp)
        for layer in self.pred_genes:
            emb = layer(emb, lat_comp)
        gene_reconstructions = self.convert_out(emb)
        return gene_reconstructions

    def update(self, drugs, cells, genes):
        gene_reconstructions  = self.predict(drugs, cells)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)

        self.optimizer_autoencoder.zero_grad()
        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2)
        self.optimizer_autoencoder.step()

        return reconstruction_loss.item()
    


class Simple_residual_feature_fine(torch.nn.Module):
    def __init__(self, pretrain_model: Simple_residual_feature, hparam):
        super(Simple_residual_feature_fine, self).__init__()

        self.loss_autoencoder = rrmse()
        self.device = hparam['device']
        self.model_pretrain = pretrain_model 
        # simple modifs
        self.feature_redu = torch.nn.Linear(18211*2, 918*2)
        self.convert_out_2 = torch.nn.Linear(918, hparam["nb_feat"])
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.model_pretrain.convert_in, False)
            + get_params(self.model_pretrain.pred_genes, False)
            + get_params(self.model_pretrain.drug_embeddings, True)
            + get_params(self.model_pretrain.cell_embeddings, True)
            + get_params(self.model_pretrain.convert_out, True)
            + get_params(self.convert_out_2, True)
            + get_params(self.feature_redu, True)
        )
        self.optimizer_autoencoder = torch.optim.Adam(_parameters, lr = hparam['lr'], weight_decay = hparam['wd'])

    def predict(self, drugs, cells):
        drugs = self.feature_redu(drugs)
        cells = self.feature_redu(cells)
        latent_drug = self.model_pretrain.drug_embeddings(drugs)
        latent_cell = self.model_pretrain.cell_embeddings(cells)
        lat_comp = tcat([latent_drug, latent_cell], dim=1)
        emb = self.model_pretrain.convert_in(lat_comp)
        for layer in self.model_pretrain.pred_genes:
            emb = layer(emb, lat_comp)
        gene_reconstructions = self.model_pretrain.convert_out(emb)
        gene_reconstructions_ = self.convert_out_2(gene_reconstructions)
        return gene_reconstructions_

    def update(self, drugs, cells, genes):
        gene_reconstructions  = self.predict(drugs, cells)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)

        self.optimizer_autoencoder.zero_grad()
        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2)
        self.optimizer_autoencoder.step()

        return reconstruction_loss.item()
    



class simple_model_feature_fine(nn.Module):
    def __init__(self, model_pretrain: simple_model_feature,hparam):
        super(simple_model_feature_fine, self).__init__()
        self.model_pretrain = model_pretrain 
        self.hparam = hparam 
        self.device = self.hparam['device']
        dim_hid = hparam['hid_dim']
        self.feature_redu = torch.nn.Linear(18211*2, 918*2)
        self.convert_out_2 = torch.nn.Linear(918, hparam["nb_feat"])
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.model_pretrain.drug_embeddings, False)
            + get_params(self.model_pretrain.cell_embeddings, False)
            + get_params(self.model_pretrain.fit_pred.network[-1], True)
            + get_params(self.model_pretrain.fit_pred.network[:-1], False)
            + get_params(self.convert_out_2, True)
            + get_params(self.feature_redu, True)
        )
        self.loss_autoencoder = rrmse()
        self.optimizer_autoencoder = optim.Adam(_parameters, lr = self.hparam['lr'], weight_decay = self.hparam['wd'])
    
    def predict(
        self, 
        drugs, 
        cells, 
    ):
        
        drugs, cells = drugs.to(self.device), cells.to(self.device)
        drugs = self.feature_redu(drugs)
        cells = self.feature_redu(cells)
        emb_1, emb_2 = self.model_pretrain.cell_embeddings(cells), self.model_pretrain.drug_embeddings(drugs)
        emb = tcat((emb_1, emb_2), dim=1)
        gene_reconstructions = self.model_pretrain.fit_pred(emb)
        gene_reconstructions_ = self.convert_out_2(gene_reconstructions)
        return gene_reconstructions_



    def update(self, drugs, cells, genes_d):
        drugs, cells = drugs.to(self.device), cells.to(self.device)
        gene_reconstructions = self.predict(drugs, cells)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes_d)
    
        self.optimizer_autoencoder.zero_grad()

        reconstruction_loss.backward()
        self.optimizer_autoencoder.step()
       
        return reconstruction_loss.item() 
