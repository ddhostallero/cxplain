import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from gene_finding.models import load_model
from gene_finding.borda import rank_aggregate_Borda
import os
import argparse
import os

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.random.set_seed(1)
from tensorflow.python.keras.losses import mean_squared_error as loss
from cxplain import MLPModelBuilder, CXPlain



# parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--seed', default=1, help='seed')
# args = parser.parse_args() 

# SEED = int(args.seed)
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

res_dir = 'gene_finding/results/CX_ind1/'

def get_ranked_list(attr, k=None):
    attr = np.abs(attr.T)

    if k is None:
        k = len(attr)
    
    sorted_list = []

    for sample in attr.columns:
        rank = attr.sort_values(sample, ascending=False).index # highest to lowest
        sorted_list.append(rank)

    agg_genes = rank_aggregate_Borda(sorted_list, k, 'geometric_mean')

    return agg_genes


def seed_borda_of_sample_bordas(all_attr, seeds):
    """
    Aggregate across all samples then aggregate across all seeds
    """

    list_of_sample_bordas = []
    # for seed in range(10):
    #     sample_borda = get_ranked_list(attr_ind[seed])
    #     list_of_sample_bordas.append(sample_borda) 

    for seed in seeds:
        sample_borda = get_ranked_list(all_attr.loc[all_attr['seed'] == seed][dataset.hgnc])
        list_of_sample_bordas.append(sample_borda) 

    return rank_aggregate_Borda(list_of_sample_bordas, 200, 'geometric_mean')

def sample_borda_of_seed_bordas(all_attr, idx):
    """
    Aggregate across all seeds then aggregate across all samples
    """
    list_of_seed_bordas = []

    # for sample in attr_ind[0].index:
    #     sample_attr_all_seed = pd.DataFrame(index=range(10), columns=dataset.hgnc)

    #     for seed in range(10):
    #         sample_attr_all_seed.loc[seed] = attr_ind[seed].loc[sample]

    #     seed_borda = get_ranked_list(sample_attr_all_seed)
    #     list_of_seed_bordas.append(seed_borda)

    # return rank_aggregate_Borda(list_of_seed_bordas, k=200, 'geometric_mean')

    for sample in idx:
        seed_borda = get_ranked_list(all_attr.loc[all_attr['sample'] == sample][dataset.hgnc])
        list_of_seed_bordas.append(seed_borda)  

    # print(len(list_of_seed_bordas), len(list_of_seed_bordas[0]))
    return rank_aggregate_Borda(list_of_seed_bordas, 200, 'geometric_mean')


def borda_of_all_tuples(all_attr):
    return get_ranked_list(all_attr[dataset.hgnc], 200)

def find_genes_CX(drug, test_tcga_expr):
    tf.keras.backend.clear_session()
    attr_ind = []
    for seed in range(1, 11):
        exp = CXPlain.load('gene_finding/results/CX_ind1/%s/seed%d/explainer'%(drug,seed), relpath=True)
        attr = exp.explain(test_tcga_expr.values)
        attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=dataset.hgnc)
        attr_ind.append(attr)


    all_attr = pd.DataFrame(columns=['seed','sample']+list(dataset.hgnc))

    i = 0
    for sample in attr_ind[0].index:
        for seed in range(10):
            all_attr.loc[i] = [seed, sample] + list(attr_ind[seed].loc[sample])
            i+=1

    print('boat')
    boat = borda_of_all_tuples(all_attr)
    print('seed then sample')
    seed_then_sample = sample_borda_of_seed_bordas(all_attr, test_tcga_expr.index)
    print('sample then seed')
    sample_then_seed = seed_borda_of_sample_bordas(all_attr, range(10))

    return boat, seed_then_sample, sample_then_seed

def find_genes(n_seeds=10):

    if not os.path.exists('gene_finding/results/CX_ind1/aggregated/'):
        os.mkdir('gene_finding/results/CX_ind1/aggregated/')

    writer_1 = pd.ExcelWriter('gene_finding/results/CX_ind1/aggregated/borda_of_all_tuples.xlsx', engine='xlsxwriter')
    writer_2 = pd.ExcelWriter('gene_finding/results/CX_ind1/aggregated/borda_of_borda_across_seeds.xlsx', engine='xlsxwriter')
    writer_3 = pd.ExcelWriter('gene_finding/results/CX_ind1/aggregated/borda_of_borda_across_sample.xlsx', engine='xlsxwriter')


    for drug in drugs:
        print('============================')
        print(drug)
        _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug, load_normalizer=True)
        boat, seed_then_sample, sample_then_seed = find_genes_CX(drug, test_tcga_expr)

        pd.DataFrame(boat, index=range(200), columns=[1]).to_excel(writer_1, sheet_name=drug, index=False)  
        pd.DataFrame(seed_then_sample, index=range(200), columns=[1]).to_excel(writer_2, sheet_name=drug, index=False)  
        pd.DataFrame(sample_then_seed, index=range(200), columns=[1]).to_excel(writer_3, sheet_name=drug, index=False)  

    writer_1.save()
    writer_2.save()
    writer_3.save()

find_genes()