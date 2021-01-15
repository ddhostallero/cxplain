import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from jupyter_utils import AllDataset


data_dir = '../drp-data/'

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

folder = 'Granger'

attr_dict = {}
for i, drug in enumerate(drugs):
    print(drug)
    # _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug)
    # exp = CXPlain.load('gene_finding/results/%s/%s/explainer'%(folder, drug), custom_model_loader=None, relpath=True)
    # attr,_ = exp.explain(test_tcga_expr.values)
    # attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=dataset.genes)
    attr = pd.read_csv('gene_finding/results/%s/%s/explainer/all_attributions.csv'%(folder, drug), index_col=0)
    attr_dict[drug]=attr

fig, axes = plt.subplots(7, 2, figsize=(14, 35))

writer_a = pd.ExcelWriter('gene_finding/results/%s/top_genes_mean_aggregation.xlsx'%folder, engine='xlsxwriter')


GDSC_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/gdsc_rma_gene_expr.csv'
genes = pd.read_csv(data_dir + GDSC_GENE_EXPRESSION, index_col=0).index

# conv = pd.DataFrame(index=genes, columns=['hgnc'])
conv = pd.DataFrame(index=attr_dict[drugs[0]].columns, columns=['ensembl'])
conv['ensembl'] = genes

for i, drug in enumerate(drugs):
    ax = axes[i%7][i//7]


    attr_drug = attr_dict[drug].mean(axis=0).sort_values(ascending=False) # sort mean attribution
    attr_drug = attr_drug/attr_drug.max()								  # normalize
    
    # find knee
    kneedle = KneeLocator(np.arange(len(attr_drug)), attr_drug, curve='convex', direction='decreasing')
    thresh = kneedle.knee

    df = pd.DataFrame(index=range(1,thresh+1), columns=['ensembl', 'hgnc'])
    df['hgnc'] = attr_drug.index[:thresh]
    df['ensembl'] = conv.loc[attr_drug.index[:thresh]]['ensembl'].values
    df.to_excel(writer_a, sheet_name=drug) 

    ax.plot(range(len(attr_drug)), attr_drug)
    ax.set_title("%s (%d)"%(drug, thresh))
    ax.set_ylabel('attribution')
    ax.axvline(thresh)

plt.savefig('gene_finding/results/%s/kneedle_thresholds.pdf'%folder)
writer_a.save()


