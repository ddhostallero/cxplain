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

folder = 'CX_ens10'

pathway_file = '../drp-data/pathways/9606.enrichr_pathway.edge'
pathway = pd.read_csv(pathway_file, sep='\t', header=None)
pathway = pathway.loc[pathway[1].isin(dataset.genes)]

gsc_filtered = '../KnowEng_GSC/GSC_10mod/drawr_filtered/DraWR_GSC_Enrichr_STRINGExp.xlsx'

attr_dict = {}
for i, drug in enumerate(drugs):
    print(drug)
    _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug)
    exp = CXPlain.load('gene_finding/results/%s/%s/explainer'%(folder, drug), custom_model_loader=None, relpath=True)
    attr,_ = exp.explain(test_tcga_expr.values)
    attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=dataset.genes)
    attr_dict[drug]=attr

# fig, axes = plt.subplots(7, 2, figsize=(14, 35))

writer_a = pd.ExcelWriter('gene_finding/results/%s/top_genes_mean_aggregation_info.xlsx'%folder, engine='xlsxwriter')

conv = pd.DataFrame(index=dataset.genes, columns=['hgnc'])
conv['hgnc'] = dataset.hgnc



for i, drug in enumerate(drugs):
    # ax = axes[i%7][i//7]


    attr_drug = attr_dict[drug].mean(axis=0).sort_values(ascending=False) # sort mean attribution
    attr_drug = attr_drug/attr_drug.max()								  # normalize
    
    # find knee
    kneedle = KneeLocator(np.arange(len(attr_drug)), attr_drug, curve='convex', direction='decreasing')
    thresh = kneedle.knee
    names = attr_drug.index[:thresh]


    top_pathways = pd.read_excel(gsc_filtered, sheet_name=drug)
    # for path_names in top_pathways['property_gene_set_id'].unique():
        # pathway.loc[pathway[0] == ]
    
    top_pathways = pathway.loc[pathway[0].isin(top_pathways['property_gene_set_id'].unique())]


    pathways_per_gene = []
    # print(top_pathways)
    # print(names)
    for gene in names:
        gene_path = top_pathways.loc[top_pathways[1] == gene][0].unique()
        if len(gene_path) == 0:
            pathways_per_gene.append("")
        else:
            pathways_per_gene.append(';'.join(gene_path))

    df = pd.DataFrame(index=range(1,thresh+1))
    df['ensembl'] = names
    df['hgnc'] = conv.loc[names]['hgnc'].values
    df['attribution'] = attr_drug[names].values
    df['pathways'] = pathways_per_gene

    df.to_excel(writer_a, sheet_name=drug) 

    # ax.plot(range(len(dataset.hgnc)), attr_drug)
    # ax.set_title("%s (%d)"%(drug, thresh))
    # ax.set_ylabel('attribution')
    # ax.axvline(thresh)

# plt.savefig('gene_finding/results/%s/kneedle_thresholds.pdf'%folder)
writer_a.save()


