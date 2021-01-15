import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from jupyter_utils import AllDataset
from gene_finding.borda import rank_aggregate_Borda

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.random.set_seed(1)
from cxplain import CXPlain

# mode = 'mean_of_means'
mode = 'borda_of_means'

data_dir = '../drp-data/'
GDSC_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/gdsc_rma_gene_expr.csv'
TCGA_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/tcga_log2_gene_expr.csv'

TCGA_TISSUE = 'preprocessed/tissue_type/TCGA_tissue_one_hot.csv'
GDSC_TISSUE = 'preprocessed/tissue_type/GDSC_tissue_one_hot.csv'

GDSC_lnIC50 = 'preprocessed/drug_response/gdsc_lnic50.csv'
TCGA_DR = 'preprocessed/drug_response/tcga_drug_response.csv'

dataset = AllDataset(data_dir, GDSC_GENE_EXPRESSION, TCGA_GENE_EXPRESSION, 
                     GDSC_lnIC50, TCGA_DR, TCGA_TISSUE)

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

folder = 'CX_ens1'

attr_dict = {}
for i, drug in enumerate(drugs):
    print(drug)
    _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug)
    exp = CXPlain.load('gene_finding/results/%s/%s/explainer'%(folder, drug), custom_model_loader=None, relpath=True)
    attr = exp.explain(test_tcga_expr.values)
    attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=dataset.genes)
    attr_dict[drug]=attr

# writer_a = pd.ExcelWriter('gene_finding/results/CX_ens1/top_genes_mean_aggregation.xlsx', engine='xlsxwriter')

conv = pd.DataFrame(index=dataset.genes, columns=['hgnc'])
conv['hgnc'] = dataset.hgnc

def mean_of_means(means):
    return means.mean(axis=1).sort_values(ascending=False)    

def borda_of_means(means):
    sorted_list = []

    for sample in means.columns:
        rank = means.sort_values(sample, ascending=False).index # highest to lowest
        sorted_list.append(rank)

    agg_genes = rank_aggregate_Borda(sorted_list, len(dataset.genes), 'geometric_mean')
    return agg_genes


means = pd.DataFrame(index=dataset.genes, columns=drugs)
for drug in drugs:
    means[drug] = attr_dict[drug].mean(axis=0)

if mode == 'mean_of_means':
    m = mean_of_means(means)
    kneedle = KneeLocator(np.arange(len(m)), m, curve='convex', direction='decreasing')
    thresh = kneedle.knee
    genes = m.index[:thresh]
    plt.plot(range(len(m)), m)
    plt.axvline(thresh)
    plt.savefig('gene_finding/results/%s/kneedle_agg.pdf'%folder)
    plt.show()
else:
    m = borda_of_means(means)
    genes = m[:200]
    thresh = 200

df = pd.DataFrame(index=range(1, thresh+1), columns=['ensembl', 'hgnc'])
df['ensembl'] = genes
df['hgnc'] = conv.loc[genes]['hgnc'].values
df.to_csv('gene_finding/results/%s/top_genes_%s.csv'%(folder, mode))

