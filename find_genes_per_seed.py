import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from gene_finding.models import load_model, EnsModel
from gene_finding.borda import rank_aggregate_Borda
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', default=1, help='seed')
args = parser.parse_args() 

SEED = int(args.seed)
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
    'irinotecan',]
    # 'oxaliplatin',
    # 'paclitaxel',
    # 'pemetrexed',
    # 'tamoxifen',
    # 'temozolomide',
    # 'vinorelbine']
    
dataset = AllDataset(data_dir, GDSC_GENE_EXPRESSION, TCGA_GENE_EXPRESSION, 
                     GDSC_lnIC50, TCGA_DR, TCGA_TISSUE)

res_dir = 'gene_finding/results/'
explainer = 'CX'

# res_dir = res_dir + explainer + '/'
res_dir = res_dir + 'CX_ind1/'


def boxplots(save_dir, meta, outputs):
    ctg = ["Complete Response", "Partial Response", "Stable Disease", "Clinical Progressive Disease"]
    response = ['CR', 'PR', 'SD', 'CPD']

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    
    for i in range(1, 11):
        ax = axes[(i-1)//5][(i-1)%5]

        boxes = []
        for c in ctg:
            x = meta.loc[meta['label'] == c].index
            boxes.append(outputs.loc[x][i])

        ax.boxplot(boxes)    

        for j, box in enumerate(boxes):
            y = box.values
            x = np.random.normal(j+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.7)

        ax.set_xticklabels(response)
        ax.set_title('seed = %d'%i)

    plt.savefig(save_dir + '/classes.png')


def attribute(att_func, x, baselines, x_idx):

  return pd.DataFrame(att_func.attribute(x, baselines=baselines).detach().numpy(), 
      index=x_idx, columns=dataset.hgnc)


def get_ranked_list(attr, k=200):
    attr = np.abs(attr.T)
    sorted_list = []

    for sample in attr.columns:
        rank = attr.sort_values(sample, ascending=False).index # highest to lowest
        sorted_list.append(rank)

    agg_genes = rank_aggregate_Borda(sorted_list, k, 'geometric_mean')

    return agg_genes


def find_genes_GSClass(drug, ens, meta, test_tcga_expr):
    from captum.attr import GradientShap
    gs = GradientShap(ens)


    # find genes for Sensitive Class
    sensitive = ['Complete Response', 'Partial Response']

    sen_idx = meta.loc[meta['label'].isin(sensitive)].index
    sen = torch.FloatTensor(test_tcga_expr.loc[sen_idx].values)

    res_idx = meta.loc[~meta['label'].isin(sensitive)].index
    res = torch.FloatTensor(test_tcga_expr.loc[res_idx].values)

    sen_attr = attribute(gs, sen, res, sen_idx)
    res_attr = attribute(gs, res, sen, res_idx)

    sen_genes = get_ranked_list(sen_attr)
    res_genes = get_ranked_list(res_attr)

    out = pd.DataFrame(columns=['sensitive', 'resistant'])
    out['sensitive'] = sen_genes
    out['resistant'] = res_genes

    out.to_csv(res_dir + drug + '/genes.csv', index=False)

def find_genes_DeepLift(drug, ens, meta, test_tcga_expr):
    from captum.attr import DeepLift
    dl = DeepLift(ens)

    x = torch.FloatTensor(test_tcga_expr.values)
    attr = pd.DataFrame(dl.attribute(x).detach().numpy(), 
      index=test_tcga_expr.index, columns=dataset.hgnc)

    genes = get_ranked_list(attr)

    out = pd.DataFrame(columns=['borda', 'mean'])
    attr_mean = list(np.abs(attr).mean(axis=0).nlargest(200).index)
    out['borda'] = genes
    out['mean'] = attr_mean

    out.to_csv(res_dir + drug + '/genes.csv', index=False)

    fig, ax = plt.subplots(figsize=(15,5))
    for i in attr.index():
        ax.plt(range(len(dataset.hgnc)), attr.loc[i], alpha=0.2)
    plt.show()


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
        masked_sample = sample*mask
        data = torch.utils.data.TensorDataset(masked_sample)
        data = torch.utils.data.DataLoader(data, batch_size=2048, shuffle=False)
        
        ret_val = []
        for [x] in data:
            x = x.to(device)
            ret_val.append(model(x))

        ret_val = torch.cat(ret_val, axis=0).unsqueeze(0).cpu().detach().numpy()
        list_of_masked_outs.append(ret_val)

    masked_outs = np.concatenate(list_of_masked_outs)
    return (gdsc_expr.values, y_pred, masked_outs)


def find_genes_CX(drug, model, meta, gdsc_expr, gdsc_dr, test_tcga_expr, save_dir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    print('obtaining masked data...')
    masked_data = get_masked_data_for_CXPlain(model, gdsc_expr)
    print('obtained masked data...')
    # get_masked_data_for_CXPlain(model, test_tcga_expr)

    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    from tensorflow.python.keras.losses import mean_squared_error as loss
    from cxplain import MLPModelBuilder, CXPlain
    # from cxplain.backend.masking.zero_masking import FastZeroMasking
    model_builder = MLPModelBuilder(num_layers=2, num_units=512, batch_size=8, learning_rate=0.001)
    # masking_operation = FastZeroMasking()

    print(gdsc_expr.values.shape, gdsc_dr.values.shape)

    print("Fitting CXPlain model")
    explainer = CXPlain(model, model_builder, None, loss)
    explainer.fit(gdsc_expr.values, gdsc_dr.values, masked_data=masked_data)
    print("Attributing using CXPlain")

    attr = explainer.explain(test_tcga_expr.values)
    attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=dataset.hgnc)
    borda = get_ranked_list(attr)

    attr_mean = list(np.abs(attr).mean(axis=0).nlargest(200).index)
    out = pd.DataFrame(columns=['borda', 'mean'])
    out['borda'] = borda 
    out['mean'] = attr_mean

    out.to_csv(save_dir + '/genes.csv', index=False)

    if not os.path.exists(save_dir + '/explainer/'):
        os.mkdir(save_dir + '/explainer/')

    explainer.save(save_dir + '/explainer/')

def find_genes(drug, n_seeds=10):

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(res_dir + drug):
        os.mkdir(res_dir + drug)

    # exit()

    gdsc_expr, gdsc_dr, _, test_tcga_expr = dataset.filter_and_normalize_data(drug, load_normalizer=True)

    models = []
    outputs = pd.DataFrame(index=test_tcga_expr.index, columns=range(1,n_seeds+1))
    x = torch.FloatTensor(test_tcga_expr.values)

    print('x', x.shape)

    for i in range(1, n_seeds+1):
        mod = load_model(seed=i,drug=drug,n_genes=len(dataset.genes))
        mod.eval()
        models.append(mod)
        outputs[i] = mod(x).detach().numpy()

    fig, ax = plt.subplots()
    outputs[range(1, 11)].T.boxplot(vert=False)
    plt.tight_layout()
    plt.savefig(res_dir + drug + '/outputs.png')

    # ----

    test_tissue = dataset.tcga_tissue.loc[test_tcga_expr.index]
    tissue_list = [t for t in test_tissue.columns if test_tissue[t].sum() > 0]
    test_tissue = test_tissue[tissue_list]

    meta = pd.DataFrame(index=test_tcga_expr.index, columns=['tissue', 'label'])
    for tissue in test_tissue.columns:
        x = test_tissue.loc[test_tissue[tissue] == 1].index
        meta.loc[x, 'tissue'] = tissue

    meta['label'] = dataset.tcga_dr.loc[test_tcga_expr.index][drug]
    boxplots(res_dir + drug, meta, outputs)

    
    # ---
    # x = torch.FloatTensor(test_tcga_expr.values)
    # print(models[0](x))
    # ens = EnsModel(models)

    # print(ens(x))
    # exit()

    if explainer == 'GS':
        find_genes_GSClass(drug, ens, meta, test_tcga_expr)
    elif explainer == 'DeepLift':
        find_genes_DeepLift(drug, ens, meta, test_tcga_expr)
    elif explainer == 'CX':

        # for i in range(1, n_seeds+1):
        i = SEED-1
        save_dir = res_dir + drug + '/seed%d/'%SEED
        print(save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        find_genes_CX(drug, models[i], meta, gdsc_expr, gdsc_dr, test_tcga_expr, save_dir)    


for drug in drugs:
    print('============================')
    print(drug)
    find_genes(drug)