import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from jupyter_utils import AllDataset

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.random.set_seed(1)
from cxplain import CXPlain


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

folder = 'CX_ind1'

# attr_dict = {}
# for i, drug in enumerate(drugs):
#     print(drug)
#     _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug)
#     exp = CXPlain.load('gene_finding/results/%s/%s/explainer'%(folder, drug), custom_model_loader=None, relpath=True)
#     attr,_ = exp.explain(test_tcga_expr.values)
#     attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=dataset.genes)
#     attr_dict[drug]=attr

attr_dict = {}
conf_dict = {}
for i, drug in enumerate(drugs):
    print(drug)
    _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug)
    
    attr_all = np.zeros((len(test_tcga_expr.index), len(dataset.genes)))

    for seed in range(1, 11):
        exp = CXPlain.load('gene_finding/results/%s/%s/seed%d/explainer'%(folder, drug, seed), custom_model_loader=None, relpath=True)
        attr = exp.explain(test_tcga_expr.values)
        attr_all += attr

    
    attr = pd.DataFrame(attr_all/10.0, index=test_tcga_expr.index, columns=dataset.genes)        
    attr_dict[drug]=attr

fig, axes = plt.subplots(7, 2, figsize=(14, 35))

writer_a = pd.ExcelWriter('gene_finding/results/%s/top_genes_mean_of_means_aggregation.xlsx'%folder, engine='xlsxwriter')

conv = pd.DataFrame(index=dataset.genes, columns=['hgnc'])
conv['hgnc'] = dataset.hgnc

for i, drug in enumerate(drugs):
    ax = axes[i%7][i//7]


    attr_drug = attr_dict[drug].mean(axis=0).sort_values(ascending=False) # sort mean attribution
    attr_drug = attr_drug/attr_drug.max()								  # normalize
    
    # find knee
    kneedle = KneeLocator(np.arange(len(attr_drug)), attr_drug, curve='convex', direction='decreasing')
    thresh = kneedle.knee

    df = pd.DataFrame(index=range(1,thresh+1), columns=['ensembl', 'hgnc'])
    df['ensembl'] = attr_drug.index[:thresh]
    df['hgnc'] = conv.loc[attr_drug.index[:thresh]]['hgnc'].values
    df.to_excel(writer_a, sheet_name=drug) 

    ax.plot(range(len(dataset.hgnc)), attr_drug)
    ax.set_title("%s (%d)"%(drug, thresh))
    ax.set_ylabel('attribution')
    ax.axvline(thresh)

plt.savefig('gene_finding/results/%s/kneedle_thresholds.pdf'%folder)
writer_a.save()


