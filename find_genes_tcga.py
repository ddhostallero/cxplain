import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from gene_finding.models import load_model, EnsModel
from gene_finding.borda import rank_aggregate_Borda
import os
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--drug', default=1, help='seed')
args = parser.parse_args() 


SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

from jupyter_utils import AllDataset

data_dir = '../drp-data/'
GDSC_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/gdsc_rma_gene_expr.csv'
# TCGA_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/tcga_log2_gene_expr.csv'
TCGA_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/tcga_labeled_log2_gene_expr.csv'

TCGA_TISSUE = 'preprocessed/tissue_type/TCGA_tissue_one_hot.csv'
GDSC_TISSUE = 'preprocessed/tissue_type/GDSC_tissue_one_hot.csv'

GDSC_lnIC50 = 'preprocessed/drug_response/gdsc_lnic50.csv'
TCGA_DR = 'preprocessed/drug_response/tcga_drug_response.csv'

drugs = [
    'bleomycin',
    'cisplatin',
    'cyclophosphamide',
    'docetaxel',
    'doxorubicin',
    'etoposide',
    'gemcitabine',
    'irinotecan',
    'oxaliplatin',
    'paclitaxel',
    'pemetrexed',
    'tamoxifen',
    'temozolomide',
    'vinorelbine']
    
dataset = AllDataset(data_dir, GDSC_GENE_EXPRESSION, TCGA_GENE_EXPRESSION, 
                     GDSC_lnIC50, TCGA_DR, TCGA_TISSUE)

res_dir = 'gene_finding/results/'
explainer = 'CX'

# res_dir = res_dir + explainer + '/'
res_dir = res_dir + 'Granger/'


def get_ranked_list(attr, k=200):
    attr = np.abs(attr.T)
    sorted_list = []

    for sample in attr.columns:
        rank = attr.sort_values(sample, ascending=False).index # highest to lowest
        sorted_list.append(rank)

    agg_genes = rank_aggregate_Borda(sorted_list, k, 'geometric_mean')

    return agg_genes


def get_masked_data_for_CXPlain(model, gdsc_expr):
    x_train = torch.FloatTensor(gdsc_expr.values)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if type(model) == EnsModel:
        for m in model.model_list:
            m.to(device)

    model.to(device)
    model.eval()

    y_pred = model(x_train.to(device)).cpu().detach().numpy()
    n_genes = x_train.shape[1]

    mask = torch.ones((n_genes, n_genes)) - torch.eye(n_genes)

    list_of_masked_outs = []
    for i, sample in enumerate(x_train):
        if (i+1) % 100 == 0: print(i)
        masked_sample = sample*mask
        data = torch.utils.data.TensorDataset(masked_sample)
        data = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False)
        
        ret_val = []
        for [x] in data:
            x = x.to(device)
            ret_val.append(model(x))

        ret_val = torch.cat(ret_val, axis=0).unsqueeze(0).cpu().detach().numpy()
        list_of_masked_outs.append(ret_val)

    masked_outs = np.concatenate(list_of_masked_outs)
    return (gdsc_expr.values, y_pred, masked_outs)


def find_genes_granger(drug, model, test_tcga_expr, tcga_dr):
    print('obtaining masked data...')
    _,y_pred,masked_data = get_masked_data_for_CXPlain(model, test_tcga_expr)
    print('obtained masked data...')

    n_genes = test_tcga_expr.shape[1]

    y_pred = torch.Tensor(y_pred)
    masked_data = torch.Tensor(masked_data)
    y_true = torch.Tensor(tcga_dr).view(-1,1)

    y_pred_scaled = (y_pred - y_pred.min())/(y_pred - y_pred.min()+1e-7).max()
    masked_data = torch.relu((masked_data - y_pred.min())/(masked_data - y_pred.min() + 1e-7).max())
    y_true_tiled = y_true.repeat((1, n_genes)).view(-1, n_genes, 1)

    print(masked_data.shape, y_true.shape)

    error_with_all_feature = F.binary_cross_entropy(y_pred_scaled, y_true, reduction='none')
    error_without_one_feature = F.binary_cross_entropy(masked_data, y_true_tiled, reduction='none').view(-1, n_genes)
    delta_errors = torch.max(error_without_one_feature - error_with_all_feature, torch.ones((1, n_genes))*1e-7)
    attr = delta_errors/torch.sum(delta_errors, axis=1).view(-1, 1)

    # attr,conf = explainer.explain(test_tcga_expr.values)
    attr = pd.DataFrame(attr.cpu().detach().numpy(), index=test_tcga_expr.index, columns=dataset.hgnc)
    borda = get_ranked_list(attr)

    attr_mean = list(np.abs(attr).mean(axis=0).nlargest(200).index)
    out = pd.DataFrame(columns=['borda', 'mean'])
    out['borda'] = borda 
    out['mean'] = attr_mean

    out.to_csv(res_dir + drug + '/genes.csv', index=False)

    if not os.path.exists(res_dir + drug + '/explainer/'):
        os.mkdir(res_dir + drug + '/explainer/')
    attr.to_csv(res_dir + drug + '/explainer/all_attributions.csv')

def get_tcga_dr_bin(drug):
    x = dataset.tcga_dr[drug].dropna()
    y1 = x.isin(['Clinical Progressive Disease', 'Stable Disease'])
    return y1.values*1

def find_genes(drug, n_seeds=10):

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(res_dir + drug):
        os.mkdir(res_dir + drug)

    _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug, load_normalizer=True)

    models = []
    outputs = pd.DataFrame(index=test_tcga_expr.index, columns=range(1,n_seeds+1))
    x = torch.FloatTensor(test_tcga_expr.values)

    print('x', x.shape)

    for i in range(1, n_seeds+1):
        mod = load_model(seed=i,drug=drug,n_genes=len(dataset.genes))
        mod.eval()
        models.append(mod)
        outputs[i] = mod(x).detach().numpy()

    ens = EnsModel(models)

    tcga_dr = get_tcga_dr_bin(drug)
    find_genes_granger(drug, ens, test_tcga_expr, tcga_dr)
    


for drug in drugs:
    print('============================')
    print(drug)
    find_genes(drug)

# drug = args.drug
# find_genes(drug)