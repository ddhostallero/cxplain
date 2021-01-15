import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from jupyter_utils import AllDataset

data_dir = '../drp-data/'
GDSC_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/gdsc_rma_gene_expr.csv'
TCGA_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/tcga_log2_gene_expr.csv'

TCGA_CANCER = 'preprocessed/cancer_type/TCGA_cancer_one_hot.csv'
GDSC_CANCER = 'preprocessed/cancer_type/GDSC_cancer_one_hot.csv'

GDSC_lnIC50 = 'preprocessed/drug_response/gdsc_lnic50.csv'
TCGA_DR = 'preprocessed/drug_response/tcga_drug_response.csv'


gdsc_dr = pd.read_csv(data_dir + GDSC_lnIC50, index_col=0)
tcga_dr = pd.read_csv(data_dir + TCGA_DR, index_col=0)
gdsc_cancer = pd.read_csv(data_dir + GDSC_CANCER, index_col=0)
tcga_cancer = pd.read_csv(data_dir + TCGA_CANCER, index_col=0)






# dataset = AllDataset(data_dir, GDSC_GENE_EXPRESSION, TCGA_GENE_EXPRESSION, 
#                      GDSC_lnIC50, TCGA_DR, TCGA_TISSUE)


# cancer_typetcga =

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


writer_a = pd.ExcelWriter('gene_finding/sample_counts.xlsx', engine='xlsxwriter')

gdsc_df = pd.DataFrame(columns=drugs)
tcga_df = pd.DataFrame(columns=drugs)

for drug in drugs:
   samples_with_label = gdsc_dr.loc[drug].dropna().index
   ctype = gdsc_cancer[samples_with_label]
   gdsc_df[drug] = ctype.sum(axis=1)

   samples_with_label = tcga_dr.loc[drug].dropna().index
   ctype = tcga_cancer[samples_with_label]
   tcga_df[drug] = ctype.sum(axis=1)

tcga_df = tcga_df.sort_index()
gdsc_df = gdsc_df.sort_index()


tcga_df.to_excel(writer_a, sheet_name='TCGA (patients)') 
gdsc_df.to_excel(writer_a, sheet_name='GDSC (cell lines)') 
writer_a.save()
